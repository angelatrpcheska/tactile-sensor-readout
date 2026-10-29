"""Pytest exercises for the STM32 tactile sensor's binary USB protocol.

run:
    python -m pytest -v test_tactile_protocol.py

All tests use simulated serial data: no STM32 or sensor is needed.
"""

import struct

import numpy as np
import threading
import pytest

import read_tactile_threaded as reader


class FakeSerial:
    """Small fake of the pyserial interface used by the reader."""

    def __init__(self, data=b""):
        self.buffer = bytearray(data)

    @property
    def in_waiting(self):
        return len(self.buffer)

    def read(self, n):
        chunk = bytes(self.buffer[:n])
        del self.buffer[:n]
        return chunk

    def feed(self, data):
        self.buffer.extend(data)


def make_packet(counter, values):
    """Make a valid STM32-style packet to feed into the real Python parser."""
    values = np.asarray(values, dtype=np.uint16)
    assert values.shape == (reader.NUM_VALUES,)
    assert np.all(values <= 4095)

    # Match the C firmware: two ADC values -> three packed bytes.
    a, b = values[0::2], values[1::2]
    triples = np.empty((reader.NUM_VALUES // 2, 3), dtype=np.uint8)
    triples[:, 0] = a & 0xFF
    triples[:, 1] = ((a >> 8) & 0x0F) | ((b & 0x0F) << 4)
    triples[:, 2] = b >> 4
    payload = triples.tobytes()

    counter_bytes = struct.pack("<I", counter)
    checksum = reader.crc16_ccitt_false(counter_bytes + payload)
    packet = reader.SYNC + counter_bytes + payload + struct.pack("<H", checksum)
    assert len(packet) == reader.PACKET_SIZE
    return packet


def test_packet_dimensions():
    assert reader.NUM_ROWS == 12
    assert reader.NUM_COLS == 32
    assert reader.NUM_VALUES == 384
    assert reader.DATA_SIZE == 576
    assert reader.CRC_SIZE == 2
    assert reader.PACKET_SIZE == 584  # 2 sync + 4 counter + 576 data + 2 CRC


def test_crc_known_answer():
    # Standard independent CRC-16/CCITT-FALSE test vector.
    assert reader.crc16_ccitt_false(b"123456789") == 0x29B1


@pytest.mark.parametrize(
    "a,b",
    [(0, 0), (4095, 4095), (0xABC, 0x123), (1, 4095), (2048, 1950)]
)
def test_unpacks_two_12bit_values(a, b):
    # One known pair, followed by zero-valued ADC samples.
    first_three_bytes = bytes([
        a & 0xFF,
        ((a >> 8) & 0x0F) | ((b & 0x0F) << 4),
        (b >> 4) & 0xFF
    ])
    payload = first_three_bytes + bytes(reader.DATA_SIZE - 3)

    decoded = reader.unpack_12bit_payload(payload)
    assert decoded is not None
    assert decoded.shape == (reader.NUM_VALUES,)
    assert int(decoded[0]) == a
    assert int(decoded[1]) == b
    assert np.all(decoded[2:] == 0)


def test_wrong_payload_length_is_rejected():
    assert reader.unpack_12bit_payload(bytes(reader.DATA_SIZE - 1)) is None


def test_valid_frame_has_correct_counter_shape_and_orientation():
    values = np.zeros(reader.NUM_VALUES, dtype=np.uint16)
    values[0] = 100     # firmware [column 0][row 0]
    values[12] = 200    # firmware [column 1][row 0]
    values[11] = 300    # firmware [column 0][row 11]
    serial_port = FakeSerial(make_packet(42, values))

    counter, matrix, parsed = reader.read_latest_available_frame(serial_port, bytearray())
    assert counter == 42
    assert parsed == 1
    assert matrix.shape == (32, 12)  # firmware ordering; display transposes it
    assert matrix[0, 0] == 100
    assert matrix[1, 0] == 200
    assert matrix[0, 11] == 300


def test_corrupted_packet_is_skipped_and_next_one_recovers():
    bad = bytearray(make_packet(7, np.zeros(reader.NUM_VALUES, dtype=np.uint16)))
    bad[10] ^= 0x01  # flip one payload bit WITHOUT updating the CRC
    good_values = np.full(reader.NUM_VALUES, 1234, dtype=np.uint16)
    serial_port = FakeSerial(bytes(bad) + make_packet(8, good_values))

    counter, matrix, parsed = reader.read_latest_available_frame(serial_port, bytearray())
    assert counter == 8
    assert parsed == 1  # rejected packet did not count as valid
    assert np.all(matrix == 1234)


def test_reader_returns_newest_when_multiple_frames_are_waiting():
    serial_port = FakeSerial(
        make_packet(100, np.full(reader.NUM_VALUES, 900, dtype=np.uint16))
        + make_packet(101, np.full(reader.NUM_VALUES, 1200, dtype=np.uint16))
    )

    counter, matrix, parsed = reader.read_latest_available_frame(serial_port, bytearray())
    assert counter == 101
    assert parsed == 2
    assert np.all(matrix == 1200)


def test_partial_packet_waits_for_rest():
    packet = make_packet(56, np.full(reader.NUM_VALUES, 1950, dtype=np.uint16))
    serial_port = FakeSerial(packet[:100])
    rxbuf = bytearray()

    counter, matrix, parsed = reader.read_latest_available_frame(serial_port, rxbuf)
    assert counter is None
    assert matrix is None
    assert parsed == 0

    serial_port.feed(packet[100:])
    counter, matrix, parsed = reader.read_latest_available_frame(serial_port, rxbuf)
    assert counter == 56
    assert parsed == 1
    assert np.all(matrix == 1950)


def test_reader_discards_garbage_before_sync():
    packet = make_packet(90, np.full(reader.NUM_VALUES, 2000, dtype=np.uint16))
    serial_port = FakeSerial(b"\x01\x02\x03\xff" + packet)

    counter, matrix, parsed = reader.read_latest_available_frame(serial_port, bytearray())
    assert counter == 90
    assert parsed == 1
    assert np.all(matrix == 2000)



@pytest.mark.parametrize(
    "counters, expected_skips",
    [
        ([4294967294, 4294967295, 0, 1], 0),
        ([100, 103], 2),
        ([100, 101, 104], 2)
    ],
)
def test_counter_tracking(monkeypatch, counters, expected_skips):

    # Create a fake sensor matrix
    sample = np.full((32, 12), 1950, dtype=np.uint16)

    # Simulate frame counters arriving one at a time
    sequence = iter(counters)
    stop_event = threading.Event()

    def fake_read_latest_frame(_serial, _rxbuf):
        try:
            counter = next(sequence)
        except StopIteration:
            stop_event.set()
            return None, None, 0

        return counter, sample, 1

    # Replace the real packet-reading function
    monkeypatch.setattr(
        reader,
        "read_latest_available_frame",
        fake_read_latest_frame
    )

    # Initialize the same state as your real reader
    state = {
        "counter": None,
        "matrix": None,
        "received_frames": 0,
        "reader_dropped_frames": 0,
        "stm32_counter_skips": 0
    }

    # Execute the real counter-tracking logic
    reader.serial_reader_thread(
        None,
        state,
        threading.Lock(),
        stop_event
    )

    # Verify expected results
    assert state["counter"] == counters[-1]
    assert state["received_frames"] == len(counters)
    assert state["reader_dropped_frames"] == 0
    assert state["stm32_counter_skips"] == expected_skips

