"""Tests for tactile data acquisition using simulated serial input."""

from collections import deque
import random
import struct
import threading
import unittest
from unittest.mock import patch

import numpy as np

import check_tactile_noise as noise


def bitwise_crc_reference(data):
    """Independent, original CRC16-CCITT-FALSE bit-by-bit reference."""
    value = 0xFFFF
    for byte in data:
        value ^= byte << 8
        for _ in range(8):
            value = ((value << 1) ^ (0x1021 if value & 0x8000 else 0)) & 0xFFFF
    return value


def make_packet(counter, values=None, *, payload_format="packed12"):
    if values is None:
        values = np.arange(384, dtype=np.uint16)
    values = np.asarray(values, dtype=np.uint16).reshape(-1)
    if payload_format == "packed12":
        payload = bytearray()
        for a, b in zip(values[0::2], values[1::2]):
            a, b = int(a), int(b)
            payload.extend((a & 0xFF, ((a >> 8) & 0x0F) | ((b & 0x0F) << 4), b >> 4))
        payload = bytes(payload)
    else:
        payload = values.astype("<u2").tobytes()
    body = struct.pack("<I", counter) + payload
    return noise.SYNC + body + struct.pack("<H", bitwise_crc_reference(body))


class ControlledClock:
    def __init__(self, start=100.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


class FakeSerial:
    """Delivers queued chunks on successive polls, with an optional host read delay."""

    def __init__(self, data=b"", chunks=(), clock=None, read_delay=0.001):
        self.data = bytearray(data)
        self.chunks = deque(chunks)
        self.clock = clock
        self.read_delay = float(read_delay)
        self.timeout = 0.005
        self.reset_calls = 0
        self.read_sizes = []
        self.read_timeouts = []

    @property
    def in_waiting(self):
        if not self.data and self.chunks:
            self.data.extend(self.chunks.popleft())
        return len(self.data)

    def read(self, size):
        self.read_sizes.append(int(size))
        self.read_timeouts.append(self.timeout)
        if not self.data and self.chunks:
            self.data.extend(self.chunks.popleft())
        if self.clock is not None:
            self.clock.advance(self.read_delay)
        result = bytes(self.data[:size])
        del self.data[:size]
        return result

    def reset_input_buffer(self):
        self.reset_calls += 1
        self.data.clear()

    def feed(self, data):
        self.data.extend(data)


class ContinuousFakeSerial:
    """Thread-safe serial source with barriers after each completely drained batch."""

    def __init__(self):
        self.timeout = 0.005
        self.data = bytearray()
        self.condition = threading.Condition()
        self.idle_reads = 0
        self.read_sizes = []
        self.reset_calls = 0
        self.closed = False

    @property
    def in_waiting(self):
        with self.condition:
            return len(self.data)

    def read(self, size):
        with self.condition:
            self.read_sizes.append(int(size))
            if not self.data:
                self.idle_reads += 1
                self.condition.notify_all()
                self.condition.wait_for(lambda: self.data or self.closed, timeout=self.timeout)
            result = bytes(self.data[:size])
            del self.data[:size]
            return result

    def reset_input_buffer(self):
        with self.condition:
            self.reset_calls += 1
            self.data.clear()

    def feed_and_wait(self, data):
        """The next idle read proves the preceding frame batch was processed."""
        with self.condition:
            initial_idle_reads = self.idle_reads
            self.data.extend(data)
            self.condition.notify_all()
            if not self.condition.wait_for(lambda: self.idle_reads > initial_idle_reads, timeout=2.0):
                raise AssertionError("Receiver did not finish draining the simulated serial batch")

    def wait_until_reading(self):
        with self.condition:
            if not self.condition.wait_for(lambda: self.idle_reads > 0, timeout=2.0):
                raise AssertionError("Receiver did not start serial reception")

    def cancel_read(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def make_reader(ser, **kwargs):
    return noise.TactilePacketReader(ser, rows=12, cols=32, **kwargs)


class CRCTests(unittest.TestCase):
    def test_crc_matches_bitwise_reference_for_representative_packet_data(self):
        rng = random.Random(20261007)
        samples = [
            b"",
            b"123456789",
            bytes(580),
            bytes([255]) * 580,
            bytes(range(256)) * 3,
            make_packet(0)[2:-2],
            make_packet(0xFFFFFFFF, np.full(384, 4095))[2:-2],
            bytes(rng.randrange(256) for _ in range(580)),
        ]
        self.assertEqual(noise.TactilePacketReader.crc16_ccitt_false(b"123456789"), 0x29B1)
        for sample in samples:
            with self.subTest(length=len(sample), prefix=sample[:4]):
                self.assertEqual(
                    noise.TactilePacketReader.crc16_ccitt_false(sample),
                    bitwise_crc_reference(sample),
                )


class BatchParserTests(unittest.TestCase):
    def test_multiple_complete_packets_in_one_serial_chunk_are_all_returned(self):
        counters = [100, 101, 102, 103]
        ser = FakeSerial(b"".join(make_packet(counter) for counter in counters))
        reader = make_reader(ser)
        frames, received_counters = reader.read_available_frames()
        self.assertEqual(received_counters, counters)
        self.assertEqual(len(frames), 4)
        self.assertEqual(len(ser.read_sizes), 1)
        self.assertEqual(ser.read_sizes[0], 4 * reader.packet_size)
        self.assertEqual(reader.rxbuf, b"")
        expected = np.fliplr(np.arange(384).reshape(32, 12).T)
        for frame in frames:
            np.testing.assert_array_equal(frame, expected)

    def test_complete_packet_and_half_packet_preserve_partial_bytes(self):
        first, second = make_packet(100), make_packet(101)
        split = len(second) // 2
        ser = FakeSerial(first + second[:split])
        reader = make_reader(ser)
        frames, counters = reader.read_available_frames()
        self.assertEqual(counters, [100])
        self.assertEqual(len(frames), 1)
        self.assertEqual(reader.rxbuf, second[:split])
        ser.feed(second[split:])
        frames, counters = reader.read_available_frames()
        self.assertEqual(counters, [101])
        self.assertEqual(len(frames), 1)
        self.assertEqual(reader.rxbuf, b"")

    def test_bad_crc_packet_is_rejected_and_later_valid_packet_recovered(self):
        bad = bytearray(make_packet(101, np.zeros(384, dtype=np.uint16)))
        bad[-1] ^= 1
        ser = FakeSerial(make_packet(100) + bad + make_packet(102))
        reader = make_reader(ser)
        frames, counters = reader.read_available_frames()
        self.assertEqual(counters, [100, 102])
        self.assertEqual(len(frames), 2)
        self.assertGreaterEqual(reader.crc_errors, 1)
        self.assertEqual(reader.format_errors, 0)

    def test_invalid_adc_packet_is_rejected_without_weakening_validation(self):
        bad = make_packet(101, np.full(384, 4095, dtype=np.uint16))
        ser = FakeSerial(make_packet(100) + bad + make_packet(102))
        reader = make_reader(ser, adc_max=3000)
        frames, counters = reader.read_available_frames()
        self.assertEqual(counters, [100, 102])
        self.assertEqual(len(frames), 2)
        self.assertGreaterEqual(reader.format_errors, 1)
        self.assertEqual(reader.crc_errors, 0)

    def test_sync_split_across_chunks_recovers_packet(self):
        packet = make_packet(100)
        ser = FakeSerial(b"garbage" + packet[:1])
        reader = make_reader(ser)
        frames, counters = reader.read_available_frames()
        self.assertEqual(frames, [])
        self.assertEqual(counters, [])
        self.assertEqual(reader.rxbuf, packet[:1])
        ser.feed(packet[1:])
        frames, counters = reader.read_available_frames()
        self.assertEqual(counters, [100])
        self.assertEqual(len(frames), 1)

    def test_uint16_mode_retains_every_frame_and_orientation(self):
        values = np.arange(384, dtype=np.uint16)
        ser = FakeSerial(make_packet(100, values, payload_format="uint16") + make_packet(101, values, payload_format="uint16"))
        reader = make_reader(ser, payload_format="uint16", order="row-major", flip_lr=False)
        frames, counters = reader.read_available_frames()
        self.assertEqual(counters, [100, 101])
        for frame in frames:
            np.testing.assert_array_equal(frame, values.reshape(12, 32))

    def test_large_batch_does_not_intentionally_drop_frames(self):
        counters = list(range(100, 2100))
        ser = FakeSerial(b"".join(make_packet(counter) for counter in counters))
        reader = make_reader(ser)
        frames, received_counters = reader.read_available_frames()
        self.assertEqual(received_counters, counters)
        self.assertEqual(len(frames), 2000)
        self.assertEqual(reader.rxbuf, b"")

    def test_complete_buffered_packet_is_returned_when_next_read_times_out(self):
        ser = FakeSerial()
        reader = make_reader(ser)
        reader.rxbuf.extend(make_packet(100))
        frames, counters = reader.read_available_frames()
        self.assertEqual(counters, [100])
        self.assertEqual(len(frames), 1)
        self.assertEqual(reader.timeouts, 1)

    def test_serial_timeout_is_capped_to_deadline_and_restored_after_timeout(self):
        clock = ControlledClock()
        ser = FakeSerial(clock=clock)
        reader = make_reader(ser)
        with patch.object(noise.time, "perf_counter", clock):
            with self.assertRaises(noise.FrameTimeoutError):
                reader.read_available_frames(deadline=clock.now + 0.002)
        self.assertAlmostEqual(ser.read_timeouts[0], 0.002)
        self.assertEqual(ser.timeout, 0.005)

    def test_serial_timeout_is_restored_if_serial_read_raises(self):
        clock = ControlledClock()
        ser = FakeSerial(clock=clock)
        reader = make_reader(ser)

        def fail_read(size):
            self.assertAlmostEqual(ser.timeout, 0.002)
            raise noise.serial.SerialException("Synthetic device disconnect")

        with patch.object(noise.time, "perf_counter", clock), patch.object(ser, "read", fail_read):
            with self.assertRaises(noise.serial.SerialException):
                reader.read_available_frames(deadline=clock.now + 0.002)
        self.assertEqual(ser.timeout, 0.005)


class ContinuousReceiverTests(unittest.TestCase):
    def setUp(self):
        self.clock = ControlledClock()
        clock_patch = patch.object(noise.time, "perf_counter", self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.ser = ContinuousFakeSerial()
        self.reader = make_reader(self.ser)
        self.receiver = noise.ContinuousReceiver(self.reader)
        self.receiver.start()
        self.addCleanup(self.receiver.stop)
        self.ser.wait_until_reading()

    def feed(self, counters, values=None):
        self.ser.feed_and_wait(b"".join(make_packet(counter, values) for counter in counters))

    def finish_capture(self):
        self.clock.advance(1.0)
        return self.receiver.stop_recording()

    def test_receiver_keeps_parsing_and_discarding_while_main_thread_waits(self):
        self.feed(range(100, 110))
        first = self.receiver.snapshot_stats()["valid_frames_discarded"]
        self.feed(range(110, 120))
        second = self.receiver.snapshot_stats()["valid_frames_discarded"]
        self.assertEqual(first, 10)
        self.assertEqual(second, 20)
        self.assertEqual(self.ser.reset_calls, 0)

    def test_discard_frames_are_excluded_from_measurement_arrays(self):
        self.feed(range(100, 105), np.full(384, 111, dtype=np.uint16))
        self.receiver.start_recording(10.0)
        self.feed(range(200, 204), np.full(384, 222, dtype=np.uint16))
        frames, _, stats = self.finish_capture()
        np.testing.assert_array_equal(stats["counters"], [200, 201, 202, 203])
        self.assertEqual(frames.shape, (4, 12, 32))
        np.testing.assert_array_equal(frames, np.full((4, 12, 32), 222))
        self.assertEqual(stats["host_discarded_valid_frames"], 0)
        self.assertEqual(stats["pre_capture_frames_discarded"], 5)
        self.assertEqual(self.ser.reset_calls, 0)

    def test_partial_packet_crossing_record_boundary_is_preserved_without_buffer_clear(self):
        packet = make_packet(200, np.full(384, 222, dtype=np.uint16))
        split = len(packet) // 2
        self.ser.feed_and_wait(packet[:split])
        self.assertEqual(self.reader.rxbuf, packet[:split])
        self.receiver.start_recording(10.0)
        self.assertEqual(self.reader.rxbuf, packet[:split])
        self.ser.feed_and_wait(packet[split:] + make_packet(201, np.full(384, 333, dtype=np.uint16)))
        frames, _, stats = self.finish_capture()
        np.testing.assert_array_equal(stats["counters"], [200, 201])
        np.testing.assert_array_equal(frames[0], np.full((12, 32), 222))
        np.testing.assert_array_equal(frames[1], np.full((12, 32), 333))
        self.assertEqual(stats["counter_skips"], 0)
        self.assertEqual(self.ser.reset_calls, 0)

    def test_discard_to_record_starts_fresh_measurement_counter_sequence(self):
        self.feed([100, 101, 102])
        start = self.receiver.start_recording(10.0)
        self.feed([1000, 1001, 1002, 1003])
        frames, fps, stats = self.finish_capture()
        self.assertEqual(len(frames), 4)
        self.assertEqual(start, 100.0)
        self.assertEqual(stats["first_counter"], 1000)
        self.assertEqual(stats["last_counter"], 1003)
        self.assertEqual(stats["counter_skips"], 0)
        self.assertEqual(stats["counter_discontinuities"], 0)
        self.assertEqual(stats["counter_span_frames"], 4)
        self.assertEqual(stats["valid_frame_percent"], 100)
        self.assertEqual(stats["actual_duration_s"], 1.0)
        self.assertEqual(fps, 4.0)

    def test_inflight_discard_batch_finishes_before_record_boundary_is_acknowledged(self):
        parsing_started = threading.Event()
        release_parser = threading.Event()
        start_waiting = threading.Event()
        start_finished = threading.Event()
        errors = []
        parse_buffer = self.reader._parse_buffer
        pause_once = [True]
        self.addCleanup(release_parser.set)

        def paused_parse(*args, **kwargs):
            if pause_once[0] and len(self.reader.rxbuf) >= self.reader.packet_size:
                pause_once[0] = False
                parsing_started.set()
                if not release_parser.wait(2.0):
                    raise AssertionError("Test did not release the inflight DISCARD parser")
            return parse_buffer(*args, **kwargs)

        def request_recording():
            try:
                self.receiver.start_recording(10.0)
                start_finished.set()
            except Exception as error:
                errors.append(error)

        starter = threading.Thread(target=request_recording)
        original_wait = threading.Condition.wait

        def observe_boundary_wait(condition, timeout=None):
            # Observe the start request's actual wait for the worker's boundary
            # acknowledgement, rather than relying on a scheduling delay.
            if threading.current_thread() is starter:
                start_waiting.set()
            return original_wait(condition, timeout)

        producer_errors = []

        def feed_old_batch():
            try:
                self.feed([100, 101], np.full(384, 111, dtype=np.uint16))
            except Exception as error:
                producer_errors.append(error)

        producer = threading.Thread(target=feed_old_batch)
        with patch.object(self.reader, "_parse_buffer", paused_parse), patch.object(threading.Condition, "wait", observe_boundary_wait):
            producer.start()
            self.assertTrue(parsing_started.wait(2.0), "Receiver did not enter the old DISCARD batch")
            starter.start()
            self.assertTrue(start_waiting.wait(2.0), "Recording start did not wait for the inflight batch")
            self.assertFalse(start_finished.is_set(), "Recording boundary was enabled before old parsing completed")
            release_parser.set()
            starter.join(2.0)
            producer.join(2.0)
        self.assertFalse(starter.is_alive())
        self.assertFalse(producer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(producer_errors, [])
        self.assertTrue(start_finished.is_set())
        self.feed([200, 201], np.full(384, 222, dtype=np.uint16))
        frames, _, stats = self.finish_capture()
        np.testing.assert_array_equal(stats["counters"], [200, 201])
        np.testing.assert_array_equal(frames, np.full((2, 12, 32), 222))
        self.assertEqual(stats["pre_capture_frames_discarded"], 2)
        self.assertEqual(stats["counter_skips"], 0)

    def test_counter_gap_within_recording_is_visible_in_discontinuity_diagnostics(self):
        self.receiver.start_recording(10.0)
        self.feed([100, 101])
        self.feed([104, 105])
        frames, _, stats = self.finish_capture()
        self.assertEqual(len(frames), 4)
        self.assertEqual(stats["counter_skips"], 2)
        self.assertEqual(stats["counter_discontinuities"], 1)
        self.assertEqual(stats["largest_counter_jump"], 3)
        self.assertEqual(stats["largest_missing_counter_gap"], 2)
        self.assertEqual(stats["counter_span_frames"], 6)
        self.assertAlmostEqual(stats["valid_frame_fraction"], 4 / 6)
        self.assertEqual(stats["host_discarded_valid_frames"], 0)

    def test_record_to_discard_keeps_receiver_live_and_measurement_unchanged(self):
        self.receiver.start_recording(10.0)
        self.feed([100, 101, 102, 103])
        frames, _, stats = self.finish_capture()
        self.feed([104, 105, 106])
        self.assertEqual(len(frames), 4)
        np.testing.assert_array_equal(stats["counters"], [100, 101, 102, 103])
        self.assertEqual(self.receiver.snapshot_stats()["valid_frames_discarded"], 3)
        self.assertEqual(self.ser.reset_calls, 0)
        # The same running receiver handles a second independent capture.
        self.receiver.start_recording(10.0)
        self.feed([1000, 1001])
        second_frames, _, second_stats = self.finish_capture()
        self.assertEqual(len(second_frames), 2)
        np.testing.assert_array_equal(second_stats["counters"], [1000, 1001])
        self.assertEqual(second_stats["counter_skips"], 0)

    def test_sequential_thousand_frame_capture_forms_one_continuous_counter_run(self):
        self.receiver.start_recording(10.0)
        self.feed(range(1000, 2000))
        frames, _, stats = self.finish_capture()
        self.assertEqual(frames.shape, (1000, 12, 32))
        np.testing.assert_array_equal(stats["counters"], np.arange(1000, 2000, dtype=np.uint32))
        self.assertEqual(stats["counter_skips"], 0)
        self.assertEqual(stats["counter_discontinuities"], 0)
        self.assertEqual(stats["counter_span_frames"], 1000)
        self.assertEqual(stats["valid_frame_percent"], 100)
        self.assertEqual(stats["host_discarded_valid_frames"], 0)

    def test_uint32_counter_wraparound_remains_continuous(self):
        counters = [0xFFFFFFFE, 0xFFFFFFFF, 0, 1]
        self.receiver.start_recording(10.0)
        self.feed(counters)
        frames, _, stats = self.finish_capture()
        self.assertEqual(len(frames), 4)
        np.testing.assert_array_equal(stats["counters"], counters)
        self.assertEqual(stats["counter_skips"], 0)
        self.assertEqual(stats["counter_discontinuities"], 0)
        self.assertEqual(stats["counter_span_frames"], 4)
        self.assertEqual(stats["valid_frame_percent"], 100)

    def test_crc_failures_remain_validated_and_capture_stats_exclude_pre_capture_errors(self):
        bad_before = bytearray(make_packet(90, np.zeros(384, dtype=np.uint16)))
        bad_before[-1] ^= 1
        self.ser.feed_and_wait(bytes(bad_before) + make_packet(91))
        self.assertEqual(self.receiver.snapshot_stats()["crc_errors"], 1)
        self.receiver.start_recording(10.0)
        bad_during = bytearray(make_packet(101, np.zeros(384, dtype=np.uint16)))
        bad_during[-1] ^= 1
        self.ser.feed_and_wait(make_packet(100) + bytes(bad_during) + make_packet(102))
        frames, _, stats = self.finish_capture()
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(stats["counters"], [100, 102])
        self.assertEqual(stats["crc_errors"], 1)
        self.assertEqual(stats["format_errors"], 0)
        self.assertEqual(stats["counter_skips"], 1)
        self.assertEqual(self.receiver.snapshot_stats()["crc_errors"], 2)

    def test_exact_deadline_excludes_later_frames_and_receiver_keeps_discarding(self):
        self.receiver.start_recording(1.0)
        self.clock.advance(0.5)
        self.feed([100, 101])
        self.clock.advance(0.5)
        self.feed([102, 103])
        frames, fps, stats = self.receiver.wait_for_recording()
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(stats["counters"], [100, 101])
        self.assertEqual(stats["actual_duration_s"], 1.0)
        self.assertEqual(stats["host_capture_start"], 100.0)
        self.assertEqual(stats["host_capture_end"], 101.0)
        self.assertEqual(fps, 2.0)
        self.feed([104])
        self.assertEqual(self.receiver.snapshot_stats()["valid_frames_discarded"], 3)
        self.assertEqual(stats["host_discarded_valid_frames"], 0)

    def test_idle_timeout_at_deadline_is_lifetime_event_not_capture_timeout(self):
        self.receiver.start_recording(1.0)
        self.feed([100, 101])
        final_read_started = threading.Event()
        release_final_read = threading.Event()
        original_read = self.ser.read
        intercept_once = [True]
        self.addCleanup(release_final_read.set)

        def final_idle_read(size):
            if intercept_once[0]:
                intercept_once[0] = False
                final_read_started.set()
                if not release_final_read.wait(2.0):
                    raise AssertionError("Test did not release the final idle serial read")
                return b""
            return original_read(size)

        with patch.object(self.ser, "read", final_idle_read):
            self.assertTrue(final_read_started.wait(2.0))
            # The receiver is blocked inside this read, so these parser counts
            # are stable. The independent baseline distinguishes earlier real
            # in-capture timeouts from the synthetic cutoff timeout.
            before_timeout_count = self.reader.snapshot_stats()["timeouts"]
            capture_start_timeout_count = self.receiver._capture_before["timeouts"]
            expected_capture_timeouts = before_timeout_count - capture_start_timeout_count
            self.clock.now = 101.0
            release_final_read.set()
            frames, _, stats = self.receiver.wait_for_recording()
            self.feed([200])
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(stats["counters"], [100, 101])
        self.assertEqual(stats["timeouts"], expected_capture_timeouts)
        self.assertGreaterEqual(self.receiver.snapshot_stats()["timeouts"], before_timeout_count + 1)
        self.assertEqual(stats["actual_duration_s"], 1.0)

    def test_serial_receiver_failure_propagates_and_final_shutdown_joins_thread(self):
        self.receiver.start_recording(10.0)
        self.feed([100, 101])
        failure_attempted = threading.Event()
        failure = noise.serial.SerialException("Synthetic USB disconnect")

        def failing_read(size):
            failure_attempted.set()
            raise failure

        with patch.object(self.ser, "read", failing_read):
            self.assertTrue(failure_attempted.wait(2.0))
            with self.assertRaisesRegex(RuntimeError, "Serial receiver failed") as raised:
                self.receiver.wait_for_recording()
        self.assertIs(raised.exception.__cause__, failure)
        self.receiver.stop()
        self.assertFalse(self.receiver._thread.is_alive())
        self.assertEqual(self.receiver.mode, self.receiver.DISCARD)

    def test_pre_deadline_receipts_are_retained_when_parsing_finishes_late(self):
        self.receiver.start_recording(1.0)
        self.clock.advance(0.5)
        parsing_started = threading.Event()
        release_parser = threading.Event()
        parse_buffer = self.reader._parse_buffer
        pause_once = [True]
        producer_errors = []
        self.addCleanup(release_parser.set)

        def paused_parse(*args, **kwargs):
            if pause_once[0] and len(self.reader.rxbuf) >= self.reader.packet_size:
                pause_once[0] = False
                parsing_started.set()
                if not release_parser.wait(2.0):
                    raise AssertionError("Test did not release the pre-deadline parser")
            return parse_buffer(*args, **kwargs)

        def feed_pre_deadline():
            try:
                self.feed([100, 101])
            except Exception as error:
                producer_errors.append(error)

        producer = threading.Thread(target=feed_pre_deadline)
        with patch.object(self.reader, "_parse_buffer", paused_parse):
            producer.start()
            self.assertTrue(parsing_started.wait(2.0))
            self.clock.advance(1.0)
            release_parser.set()
            producer.join(2.0)
        self.assertFalse(producer.is_alive())
        self.assertEqual(producer_errors, [])
        frames, fps, stats = self.receiver.wait_for_recording()
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(stats["counters"], [100, 101])
        self.assertEqual(stats["actual_duration_s"], 1.0)
        self.assertEqual(stats["host_capture_end"], 101.0)
        self.assertEqual(stats["receiver_finalized_at"], 101.5)
        self.assertEqual(stats["capture_stop_lag_s"], 0.5)
        self.assertEqual(fps, 2.0)
        self.feed([102])
        self.assertEqual(self.receiver.snapshot_stats()["valid_frames_discarded"], 1)

    def test_settling_discard_diagnostic_does_not_affect_measurement_completeness(self):
        self.feed([90, 91, 92])
        self.receiver.start_recording(10.0, settling_frames_discarded=2)
        self.feed([100, 101, 102])
        _, _, stats = self.finish_capture()
        self.assertEqual(stats["pre_capture_frames_discarded"], 3)
        self.assertEqual(stats["settling_frames_discarded"], 2)
        self.assertEqual(stats["counter_skips"], 0)
        self.assertEqual(stats["valid_frame_percent"], 100)


if __name__ == "__main__":
    unittest.main()
