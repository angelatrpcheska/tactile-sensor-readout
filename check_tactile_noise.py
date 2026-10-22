#!/usr/bin/env python3
"""
check_tactile_noise.py

Thesis-friendly tactile sensor noise / signal characterization.

Packet format:
    AA 55 + uint32 frame_counter + packed 12-bit ADC payload + CRC16-CCITT-FALSE

Default orientation matches read_tactile_threaded.py display:
    firmware [col][row] = 32x12 -> display [row][col] = 12x32 -> left/right flip

Typical runs:
    python check_tactile_noise.py --port /dev/cu.usbmodemXXXX --baseline_sec 60 --skip_touch --trial_name no_touch_60s_1khz --out_root thesis_noise
    python check_tactile_noise.py --port /dev/cu.usbmodemXXXX --baseline_sec 15 --touch_sec 10 --trial_name cloth_contact --out_root thesis_noise
"""

import argparse
import binascii
import csv
import json
import platform
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import serial

SYNC = b"\xAA\x55"
DEFAULT_ROWS = 12
DEFAULT_COLS = 32
DEFAULT_HEADER_SIZE = 6
DEFAULT_CRC_SIZE = 2
DEFAULT_ADC_MAX = 4095
DEFAULT_PRACTICAL_HIGH_SATURATION = 4050


class FrameTimeoutError(Exception):
    pass


class PacketFormatError(Exception):
    pass


class TactilePacketReader:
    def __init__(
        self,
        ser: serial.Serial,
        rows: int,
        cols: int,
        header_size: int = DEFAULT_HEADER_SIZE,
        crc_size: int = DEFAULT_CRC_SIZE,
        payload_format: str = "packed12",
        order: str = "col-major",
        flip_lr: bool = True,
        adc_max: int = DEFAULT_ADC_MAX,
    ):
        self.ser = ser
        self.rows = int(rows)
        self.cols = int(cols)
        self.header_size = int(header_size)
        self.crc_size = int(crc_size)
        self.payload_format = str(payload_format)
        self.order = str(order)
        self.flip_lr = bool(flip_lr)
        self.adc_max = int(adc_max)
        self.rxbuf = bytearray()

        if self.header_size < 6:
            raise ValueError("header_size must be at least 6: AA55 + uint32 frame counter")
        if self.crc_size not in (0, 2):
            raise ValueError("crc_size must be 0 or 2")
        if self.payload_format not in ("packed12", "uint16"):
            raise ValueError("payload_format must be packed12 or uint16")
        if self.order not in ("col-major", "row-major"):
            raise ValueError("order must be col-major or row-major")

        self.num_values = self.rows * self.cols
        if self.payload_format == "packed12":
            if self.num_values % 2 != 0:
                raise ValueError("packed12 requires an even number of ADC values")
            self.payload_size = (self.num_values * 3) // 2
        else:
            self.payload_size = self.num_values * 2

        self.packet_size = self.header_size + self.payload_size + self.crc_size

        self.crc_errors = 0
        self.format_errors = 0
        self.sync_bytes_discarded = 0
        self.timeouts = 0
        self.last_batch_rxbuf_bytes = 0
        self.last_read_completed_at = 0.0

    def snapshot_stats(self) -> Dict[str, int]:
        return {
            "crc_errors": int(self.crc_errors),
            "format_errors": int(self.format_errors),
            "sync_bytes_discarded": int(self.sync_bytes_discarded),
            "timeouts": int(self.timeouts),
        }

    @staticmethod
    def crc16_ccitt_false(data: bytes) -> int:
        # Same unreflected polynomial 0x1021, initial value 0xFFFF, no xor-out;
        # the standard-library C implementation avoids a Python loop per bit.
        return binascii.crc_hqx(data, 0xFFFF)

    def _read_more(self, deadline: Optional[float] = None) -> bool:
        n = self.ser.in_waiting
        original_timeout = self.ser.timeout
        timeout_changed = False
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False
            if original_timeout is None or original_timeout > remaining:
                self.ser.timeout = remaining
                timeout_changed = True
        try:
            # Drain the current OS queue in one operation, or wait for one byte
            # when idle. Never wait longer than the remaining capture interval.
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            chunk = self.ser.read(n if n > 0 else 1)
            self.last_read_completed_at = time.perf_counter()
        finally:
            if timeout_changed:
                self.ser.timeout = original_timeout
        if not chunk:
            self.timeouts += 1
            raise FrameTimeoutError("Timeout while waiting for serial data")
        self.rxbuf.extend(chunk)
        return True

    def _unpack_12bit_payload(self, payload: bytes) -> Optional[np.ndarray]:
        raw = np.frombuffer(payload, dtype=np.uint8)
        if raw.size != self.payload_size:
            return None

        triples = raw.reshape(-1, 3).astype(np.uint16)
        a = triples[:, 0] | ((triples[:, 1] & 0x0F) << 8)
        b = ((triples[:, 1] >> 4) & 0x0F) | (triples[:, 2] << 4)

        values = np.empty(self.num_values, dtype=np.uint16)
        values[0::2] = a & 0x0FFF
        values[1::2] = b & 0x0FFF
        return values

    def _decode_payload(self, payload: bytes) -> np.ndarray:
        if self.payload_format == "packed12":
            values = self._unpack_12bit_payload(payload)
            if values is None:
                raise PacketFormatError("wrong packed12 payload size")
        else:
            values = np.frombuffer(payload, dtype="<u2")
            if values.size != self.num_values:
                raise PacketFormatError("wrong uint16 payload size")

        if int(values.max()) > self.adc_max:
            raise PacketFormatError("ADC value above adc_max")
        return values

    def _values_to_matrix(self, values: np.ndarray) -> np.ndarray:
        if self.order == "col-major":
            frame = values.reshape((self.cols, self.rows)).T
        else:
            frame = values.reshape((self.rows, self.cols))

        if self.flip_lr:
            frame = np.fliplr(frame)

        # Preserve the exact 12-bit integer measurements and halve RAM use.
        return np.ascontiguousarray(frame, dtype=np.uint16)

    def _parse_buffer(self, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:
        """Decode complete packets, keeping a partial tail for the next read."""
        frames = []
        counters = []
        offset = 0
        while max_frames is None or len(frames) < max_frames:
            sync_idx = self.rxbuf.find(SYNC, offset)

            if sync_idx < 0:
                # Preserve the last byte in case AA and 55 cross read boundaries.
                discarded = max(0, len(self.rxbuf) - offset - 1)
                self.sync_bytes_discarded += discarded
                offset += discarded
                break

            self.sync_bytes_discarded += sync_idx - offset
            offset = sync_idx

            if len(self.rxbuf) - offset < self.packet_size:
                break

            packet = bytes(self.rxbuf[offset:offset + self.packet_size])
            counter_bytes = packet[2:6]
            payload_start = self.header_size
            payload_end = self.header_size + self.payload_size
            payload = packet[payload_start:payload_end]

            if self.crc_size == 2:
                crc_rx = struct.unpack("<H", packet[payload_end:payload_end + 2])[0]
                crc_calc = self.crc16_ccitt_false(counter_bytes + payload)
                if crc_rx != crc_calc:
                    self.crc_errors += 1
                    offset += 1
                    continue

            try:
                values = self._decode_payload(payload)
            except PacketFormatError:
                self.format_errors += 1
                offset += 1
                continue

            offset += self.packet_size
            counter = struct.unpack("<I", counter_bytes)[0]
            frames.append(self._values_to_matrix(values))
            counters.append(counter)

        # Compact once per batch, avoiding repeated shifts of a large RX queue.
        del self.rxbuf[:offset]
        return frames, counters

    def read_frame(self) -> Tuple[np.ndarray, int]:
        """Single-frame interface for startup/warmup only."""
        while True:
            frames, counters = self._parse_buffer(max_frames=1)
            if frames:
                return frames[0], counters[0]
            self._read_more()

    def read_available_frames(
        self, deadline: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Drain serial once and return EVERY complete valid frame in order."""
        # Already-read bytes are still parsed if a read times out. No complete
        # valid packet is intentionally discarded during the recording.
        try:
            self._read_more(deadline)
        except FrameTimeoutError:
            if len(self.rxbuf) < self.packet_size:
                raise
        self.last_batch_rxbuf_bytes = len(self.rxbuf)
        return self._parse_buffer()

def stats_delta(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
    return {k: int(after.get(k, 0) - before.get(k, 0)) for k in set(after) | set(before)}


class ContinuousReceiver:
    """Single serial owner with receiver-acknowledged DISCARD/RECORD boundaries.

    Mode changes happen between fully handled batches. Receipt is completion
    in a host read; a partial packet completed after RECORD starts is eligible.
    Both switches preserve RX bytes and lifetime diagnostics.
    """

    DISCARD = "DISCARD"
    RECORD = "RECORD"
    POLL_SECONDS = 0.02

    def __init__(self, reader: TactilePacketReader):
        self.reader = reader
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tactile-receiver", daemon=True)
        self._started = False
        self._error = None
        self._mode = self.DISCARD
        self._pending_start = None
        self._manual_stop = False
        self._capture_requested = False
        self._completed_capture = None
        self._frames = []
        self._counters = []
        self._capture_start = 0.0
        self._capture_deadline = 0.0
        self._capture_before = {}
        self._capture_stats = {}
        self._valid_frames_discarded = 0
        self._last_counter = None
        self._lifetime_counter_skips = 0
        self._lifetime_counter_discontinuities = 0
        self._published_stats = reader.snapshot_stats()

    @property
    def mode(self) -> str:
        with self._condition:
            return self._mode

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Receiver already started")
        self._started = True
        self._thread.start()

    def _check_running_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("Serial receiver failed") from self._error
        if not self._started or self._stop_event.is_set():
            raise RuntimeError("Serial receiver is not running")

    def snapshot_stats(self) -> Dict[str, int]:
        with self._condition:
            self._check_running_locked()
            return {
                **self._published_stats,
                "valid_frames_discarded": self._valid_frames_discarded,
                "lifetime_counter_skips": self._lifetime_counter_skips,
                "lifetime_counter_discontinuities": self._lifetime_counter_discontinuities,
            }

    def wait_for_discarded_frames(self, n: int, timeout_s: float = 10.0) -> Dict[str, int]:
        """Require valid startup packets; fail instead of waiting forever."""
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Warmup timeout must be finite and positive")
        before = self.snapshot_stats()
        deadline = time.perf_counter() + timeout_s
        with self._condition:
            target = before["valid_frames_discarded"] + max(0, int(n))
            while self._valid_frames_discarded < target:
                self._check_running_locked()
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    stats = self.snapshot_stats()
                    raise TimeoutError(
                        f"Only {self._valid_frames_discarded - before['valid_frames_discarded']} "
                        f"of {n} valid warmup frames received within {timeout_s:g} s. "
                        f"Check USB port, firmware packet format and CRC; "
                        f"receiver statistics: {stats}"
                    )
                self._condition.wait(timeout=remaining)
            return stats_delta(self.snapshot_stats(), before)

    def start_recording(self, duration_s: float, settling_frames_discarded: int = 0) -> float:
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Capture duration must be finite and positive")
        with self._condition:
            self._check_running_locked()
            if self._capture_requested:
                raise RuntimeError("Consume the previous capture before starting another")
            self._capture_requested = True
            self._pending_start = (float(duration_s), int(settling_frames_discarded))
            self._condition.notify_all()
            while self._pending_start is not None:
                self._check_running_locked()
                self._condition.wait()
            return self._capture_start

    def wait_for_recording(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        with self._condition:
            if not self._capture_requested:
                raise RuntimeError("No recording has been requested")
            while self._completed_capture is None:
                self._check_running_locked()
                self._condition.wait()
            frames, counters, stats = self._completed_capture
            self._completed_capture = None
            self._capture_requested = False
        # Stack/diagnose in the caller while the receiver continues draining.
        return summarize_capture(frames, counters, stats)

    def stop_recording(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        with self._condition:
            self._check_running_locked()
            if not self._capture_requested:
                raise RuntimeError("No recording has been requested")
            self._manual_stop = True
            self._condition.notify_all()
        return self.wait_for_recording()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        # Cancel only at final shutdown, then join before the port is closed.
        if self._started and self._thread.is_alive():
            cancel = getattr(self.reader.ser, "cancel_read", None)
            if cancel is not None:
                try:
                    cancel()
                except (OSError, serial.SerialException):
                    pass
            self._thread.join()

    def _start_capture_locked(self) -> None:
        duration_s, settling_discarded = self._pending_start
        self._frames = []
        self._counters = []
        self._capture_before = self.reader.snapshot_stats()
        self._capture_stats = {
            "requested_duration_s": duration_s,
            "pre_capture_frames_discarded": self._valid_frames_discarded,
            "settling_frames_discarded": settling_discarded,
            "acquisition_batches": 0,
            "maximum_frames_in_batch": 0,
            "maximum_rxbuf_bytes": 0,
            "host_discarded_valid_frames": 0,
        }
        self._capture_start = time.perf_counter()
        self._capture_deadline = self._capture_start + duration_s
        self._mode = self.RECORD
        self._manual_stop = False
        self._pending_start = None
        self._condition.notify_all()

    def _finish_capture_locked(self, cutoff: float, parser_after: Optional[Dict[str, int]] = None) -> None:
        self._mode = self.DISCARD
        finalized_at = time.perf_counter()
        stats = {
            **stats_delta(self.reader.snapshot_stats() if parser_after is None else parser_after, self._capture_before),
            **self._capture_stats,
            "capture_start_time": self._capture_start,
            "capture_stop_time": cutoff,
            "host_capture_start": self._capture_start,
            "host_capture_end": cutoff,
            "duration_s": cutoff - self._capture_start,
            "actual_duration_s": cutoff - self._capture_start,
            "receiver_finalized_at": finalized_at,
            "capture_stop_lag_s": max(0.0, finalized_at - cutoff),
            "capture_boundary_policy": "continuous DISCARD/RECORD receiver; start acknowledged between batches; packet receipt is completion in a host read; accept receipts before monotonic deadline; no capture buffer resets",
        }
        self._completed_capture = (self._frames, self._counters, stats)
        self._frames = []
        self._counters = []
        self._condition.notify_all()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                with self._condition:
                    if self._pending_start is not None:
                        self._start_capture_locked()
                    now = time.perf_counter()
                    if self._mode == self.RECORD and (self._manual_stop or now >= self._capture_deadline):
                        self._finish_capture_locked(min(now, self._capture_deadline))
                    read_deadline = now + self.POLL_SECONDS
                    if self._mode == self.RECORD:
                        read_deadline = min(read_deadline, self._capture_deadline)

                parser_before_read = self.reader.snapshot_stats()
                try:
                    self.reader._read_more(read_deadline)
                except FrameTimeoutError:
                    pass

                with self._condition:
                    # Finish BEFORE parsing late packets so their frames and
                    # CRC/format errors belong to DISCARD, not this capture.
                    if self._mode == self.RECORD and self.reader.last_read_completed_at >= self._capture_deadline:
                        # A timeout caused solely by the deadline is a lifetime
                        # event, not an acquisition timeout within this interval.
                        self._finish_capture_locked(self._capture_deadline, parser_before_read)
                    record_batch = self._mode == self.RECORD

                rxbuf_bytes = len(self.reader.rxbuf)
                frames, counters = self.reader._parse_buffer()
                with self._condition:
                    for counter in counters:
                        if self._last_counter is not None:
                            missing = (counter - self._last_counter - 1) & 0xFFFFFFFF
                            if missing:
                                self._lifetime_counter_skips += missing
                                self._lifetime_counter_discontinuities += 1
                        self._last_counter = counter
                    if record_batch:
                        self._frames.extend(frames)
                        self._counters.extend(counters)
                        if frames:
                            self._capture_stats["acquisition_batches"] += 1
                        self._capture_stats["maximum_frames_in_batch"] = max(self._capture_stats["maximum_frames_in_batch"], len(frames))
                        self._capture_stats["maximum_rxbuf_bytes"] = max(self._capture_stats["maximum_rxbuf_bytes"], rxbuf_bytes)
                    else:
                        self._valid_frames_discarded += len(frames)
                    self._published_stats = self.reader.snapshot_stats()
                    if self._mode == self.RECORD and time.perf_counter() >= self._capture_deadline:
                        # Keep predeadline receipts even if parsing finishes
                        # late. The receipt cutoff remains the logical deadline;
                        # report finalization lag separately from the interval.
                        self._finish_capture_locked(self._capture_deadline)
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._mode = self.DISCARD
                self._condition.notify_all()


def discard_fresh_frames(
    receiver: ContinuousReceiver, n: int, timeout_s: float = 10.0
) -> Dict[str, int]:
    """Wait for warmup discards, with a finite timeout."""
    return receiver.wait_for_discarded_frames(n, timeout_s)


def capture_frames(
    receiver: ContinuousReceiver, duration_s: float, settling_frames_discarded: int = 0,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    receiver.start_recording(duration_s, settling_frames_discarded)
    return receiver.wait_for_recording()


def summarize_capture(
    frames: List[np.ndarray], counters: List[int], stats: Dict[str, Any],
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Compute diagnostics in the caller after recording has ended."""
    if not frames:
        raise RuntimeError("No valid frames captured. Check port, packet format, CRC, and firmware.")

    frames_np = np.stack(frames, axis=0)
    counters_np = np.asarray(counters, dtype=np.uint32)
    counter_values = counters_np.astype(np.int64)
    steps = (counter_values[1:] - counter_values[:-1]) & 0xFFFFFFFF
    missing = (steps - 1) & 0xFFFFFFFF
    counter_skips = int(missing.sum())
    fps = len(frames_np) / max(stats["actual_duration_s"], 1e-9)
    # Inclusive uint32 span assumes no firmware reset and fewer than 2**32
    # USB-send attempts in the capture. Failed scans do not increment this counter.
    counter_span_frames = ((int(counters_np[-1]) - int(counters_np[0])) & 0xFFFFFFFF) + 1
    valid_frame_fraction = len(frames_np) / counter_span_frames

    stats = {
        **stats,
        "frames": int(len(frames_np)),
        "fps": float(fps),
        "counter_skips": counter_skips,
        "counter_discontinuities": int(np.count_nonzero(missing)),
        "largest_counter_jump": int(steps.max()) if len(steps) else 0,
        "largest_missing_counter_gap": int(missing.max()) if len(missing) else 0,
        "first_counter": int(counters_np[0]),
        "last_counter": int(counters_np[-1]),
        "counter_span_frames": counter_span_frames,
        "valid_frame_fraction": float(valid_frame_fraction),
        "valid_frame_percent": float(100.0 * valid_frame_fraction),
        "counters": counters_np,
    }
    return frames_np, fps, stats


def save_heatmap_png(
    path: Path,
    data: np.ndarray,
    title: str,
    colorbar_label: str = "ADC counts",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Save a heatmap. If vmin/vmax are given, the color scale is fixed."""
    arr = np.asarray(data, dtype=np.float32)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        im = plt.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.colorbar(im, label=colorbar_label)
        plt.title(title)
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return
    except Exception:
        pass

    try:
        import cv2
        lo = float(np.nanmin(arr)) if vmin is None else float(vmin)
        hi = float(np.nanmax(arr)) if vmax is None else float(vmax)
        if hi > lo:
            norm = (arr - lo) * 255.0 / (hi - lo)
        else:
            norm = np.zeros_like(arr)
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
        img = cv2.resize(img, (960, 360), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(path), img)
    except Exception as e:
        print(f"[WARN] Could not save heatmap {path.name}: {e}")


def matrix_summary(arr: np.ndarray, prefix: str = "") -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        f"{prefix}min": float(np.nanmin(a)),
        f"{prefix}p05": float(np.nanpercentile(a, 5)),
        f"{prefix}median": float(np.nanmedian(a)),
        f"{prefix}mean": float(np.nanmean(a)),
        f"{prefix}p95": float(np.nanpercentile(a, 95)),
        f"{prefix}max": float(np.nanmax(a)),
    }


def compute_capture_matrices(frames: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-taxel statistics; drift is second-half mean minus first-half mean."""
    # Accumulate in float64: small baseline drift must not be rounded away
    # while summing tens of thousands of ~2000-count samples.
    mean = frames.mean(axis=0, dtype=np.float64)
    median = np.median(frames, axis=0).astype(np.float64)
    std = frames.std(axis=0, dtype=np.float64)
    p2p = frames.max(axis=0).astype(np.float64) - frames.min(axis=0).astype(np.float64)

    n = frames.shape[0]
    if n >= 4:
        drift = (frames[n // 2:].mean(axis=0, dtype=np.float64)
                 - frames[:n // 2].mean(axis=0, dtype=np.float64))
    else:
        drift = np.zeros_like(mean)

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "p2p": p2p,
        "drift_second_half_minus_first_half": drift,
        "abs_drift": np.abs(drift),
    }


def print_capture_summary(frames: np.ndarray, fps: float, stats: Dict[str, Any]) -> None:
    print(f"Requested duration: {stats['requested_duration_s']:.3f} s")
    print(f"Frames captured: {len(frames)}")
    print(f"Measured valid frame rate: {fps:.2f} Hz")
    print(f"Actual host capture duration: {stats['actual_duration_s']:.3f} s")
    print(f"CRC errors rejected: {stats['crc_errors']}")
    print(f"Format errors rejected: {stats['format_errors']}")
    print(f"Serial timeouts: {stats['timeouts']}")
    print(f"Bytes discarded while resyncing: {stats['sync_bytes_discarded']}")
    print(f"STM32 counter skips: {stats['counter_skips']}")
    print(f"Counter range: {stats['first_counter']} -> {stats['last_counter']}")
    print(f"Counter discontinuities: {stats['counter_discontinuities']}")
    print(f"Largest counter step: {stats['largest_counter_jump']} | largest missing-counter gap: {stats['largest_missing_counter_gap']}")
    print(f"STM32 counter span (inclusive): {stats['counter_span_frames']} USB-send attempts")
    print(f"Valid frame fraction: {stats['valid_frame_fraction']:.6f} ({stats['valid_frame_percent']:.3f}%)")
    print(f"Host valid frames dropped: {stats['host_discarded_valid_frames']}")
    print(f"Pre-capture valid frames discarded: {stats['pre_capture_frames_discarded']}")
    print(f"Settling valid frames discarded: {stats['settling_frames_discarded']}")
    print(f"Receiver finalization after cutoff: {stats['capture_stop_lag_s']:.6f} s")
    print(f"Acquisition batches: {stats['acquisition_batches']} | maximum frames/batch: {stats['maximum_frames_in_batch']} | maximum RX buffer: {stats['maximum_rxbuf_bytes']} bytes")


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items() if k != "counters"}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_per_taxel_csv(
    path: Path,
    rows: int,
    cols: int,
    baseline: Dict[str, np.ndarray],
    adc_max: int,
    saturation_margin: int,
    touch: Optional[Dict[str, np.ndarray]] = None,
    practical_high_saturation: int = DEFAULT_PRACTICAL_HIGH_SATURATION,
) -> None:
    """Saturation flags use baseline means; near_high_saturation is the ADC-limit check."""
    header = [
        "row", "col",
        "baseline_mean_counts", "baseline_median_counts",
        "noise_std_counts", "noise_peak_to_peak_counts",
        "drift_second_half_minus_first_half_counts", "abs_drift_counts",
        "near_low_saturation", "near_high_saturation",
        "near_practical_high_saturation",
    ]
    if touch is not None:
        header += [
            "touch_mean_counts", "delta_touch_minus_baseline_counts",
            "abs_delta_counts", "snr_abs_delta_over_noise_std", "reliable_taxel",
        ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in range(rows):
            for c in range(cols):
                bmean = float(baseline["mean"][r, c])
                row = [
                    r, c,
                    bmean,
                    float(baseline["median"][r, c]),
                    float(baseline["std"][r, c]),
                    float(baseline["p2p"][r, c]),
                    float(baseline["drift_second_half_minus_first_half"][r, c]),
                    float(baseline["abs_drift"][r, c]),
                    int(bmean < saturation_margin),
                    int(bmean > adc_max - saturation_margin),
                    int(bmean >= practical_high_saturation),
                ]
                if touch is not None:
                    row += [
                        float(touch["mean"][r, c]),
                        float(touch["delta"][r, c]),
                        float(touch["abs_delta"][r, c]),
                        float(touch["snr"][r, c]),
                        int(touch["reliable"][r, c]),
                    ]
                writer.writerow(row)


def save_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    row = {
        "trial_name": summary["trial_name"],
        "timestamp": summary["timestamp"],
        "rows": summary["sensor"]["rows"],
        "cols": summary["sensor"]["cols"],
        "taxels": summary["sensor"]["taxels"],
        "payload_format": summary["packet"]["payload_format"],
        "packet_size_bytes": summary["packet"]["packet_size_bytes"],
        "baseline_frames": summary["baseline"]["frames"],
        "baseline_fps": summary["baseline"]["fps_hz"],
        "noise_std_median_counts": summary["baseline"]["noise_std_counts"]["median"],
        "noise_std_max_counts": summary["baseline"]["noise_std_counts"]["max"],
        "noise_p2p_median_counts": summary["baseline"]["noise_p2p_counts"]["median"],
        "noise_p2p_max_counts": summary["baseline"]["noise_p2p_counts"]["max"],
        "abs_drift_median_counts": summary["baseline"]["abs_drift_counts"]["median"],
        "abs_drift_max_counts": summary["baseline"]["abs_drift_counts"]["max"],
        "near_low_saturation_taxels": summary["baseline"]["near_low_saturation_taxels"],
        "near_high_saturation_taxels": summary["baseline"]["near_high_saturation_taxels"],
        "practical_high_saturation_threshold_counts": summary["sensor"]["practical_high_saturation_threshold_counts"],
        "near_practical_high_saturation_taxels": summary["baseline"]["near_practical_high_saturation_taxels"],
        "baseline_counter_skips": summary["baseline"]["parser_stats"]["counter_skips"],
        "baseline_crc_errors": summary["baseline"]["parser_stats"]["crc_errors"],
        "baseline_counter_span_frames": summary["baseline"]["parser_stats"]["counter_span_frames"],
        "baseline_valid_frame_fraction": summary["baseline"]["parser_stats"]["valid_frame_fraction"],
        "baseline_valid_frame_percent": summary["baseline"]["parser_stats"]["valid_frame_percent"],
    }

    if "touch" in summary:
        row.update({
            "touch_frames": summary["touch"]["frames"],
            "touch_fps": summary["touch"]["fps_hz"],
            "signal_abs_delta_median_counts": summary["touch"]["abs_delta_counts"]["median"],
            "signal_abs_delta_max_counts": summary["touch"]["abs_delta_counts"]["max"],
            "snr_median": summary["touch"]["snr"]["median"],
            "snr_max": summary["touch"]["snr"]["max"],
            "reliable_taxels": summary["touch"]["reliable_taxels"],
            "touch_counter_skips": summary["touch"]["parser_stats"]["counter_skips"],
            "touch_crc_errors": summary["touch"]["parser_stats"]["crc_errors"],
            "touch_counter_span_frames": summary["touch"]["parser_stats"]["counter_span_frames"],
            "touch_valid_frame_fraction": summary["touch"]["parser_stats"]["valid_frame_fraction"],
            "touch_valid_frame_percent": summary["touch"]["parser_stats"]["valid_frame_percent"],
        })

    for phase in ("baseline", "touch"):
        if phase in summary:
            for key in (
                "requested_duration_s", "actual_duration_s",
                "host_capture_start", "host_capture_end",
                "acquisition_batches", "maximum_frames_in_batch",
                "maximum_rxbuf_bytes", "host_discarded_valid_frames",
                "capture_start_time", "capture_stop_time",
                "receiver_finalized_at", "capture_stop_lag_s",
                "pre_capture_frames_discarded", "settling_frames_discarded",
                "counter_discontinuities", "largest_counter_jump",
                "largest_missing_counter_gap",
            ):
                row[f"{phase}_{key}"] = summary[phase]["parser_stats"][key]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def make_thesis_summary_text(summary: Dict[str, Any]) -> str:
    b = summary["baseline"]
    lines = [
        "TACTILE SENSOR NOISE MEASUREMENT SUMMARY",
        f"Trial: {summary['trial_name']}",
        f"Timestamp: {summary['timestamp']}",
        f"Sensor matrix: {summary['sensor']['rows']} x {summary['sensor']['cols']} = {summary['sensor']['taxels']} taxels",
        f"Packet format: {summary['packet']['payload_format']}, packet={summary['packet']['packet_size_bytes']} bytes, CRC={'on' if summary['packet']['crc_size_bytes'] else 'off'}",
        "",
        "No-touch baseline:",
        f"  requested duration: {b['parser_stats']['requested_duration_s']:.3f} s",
        f"  captured frames: {b['frames']} over {b['duration_s']:.3f} s",
        f"  valid frame rate: {b['fps_hz']:.2f} Hz",
        f"  STM32 counter span (inclusive): {b['parser_stats']['counter_span_frames']} frame attempts",
        f"  valid frame fraction: {b['parser_stats']['valid_frame_fraction']:.6f} ({b['parser_stats']['valid_frame_percent']:.3f}%)",
        f"  counter range: {b['parser_stats']['first_counter']} -> {b['parser_stats']['last_counter']}",
        f"  counter discontinuities: {b['parser_stats']['counter_discontinuities']}, largest step: {b['parser_stats']['largest_counter_jump']}, largest missing gap: {b['parser_stats']['largest_missing_counter_gap']}",
        f"  noise std median / max: {b['noise_std_counts']['median']:.2f} / {b['noise_std_counts']['max']:.2f} ADC counts",
        f"  noise peak-to-peak median / max: {b['noise_p2p_counts']['median']:.2f} / {b['noise_p2p_counts']['max']:.2f} ADC counts",
        f"  absolute drift median / max: {b['abs_drift_counts']['median']:.2f} / {b['abs_drift_counts']['max']:.2f} ADC counts",
        f"  near ADC-limit saturation taxels low/high (baseline mean): {b['near_low_saturation_taxels']} / {b['near_high_saturation_taxels']} (low < {summary['sensor']['saturation_margin_counts']}, high > {summary['sensor']['adc_max'] - summary['sensor']['saturation_margin_counts']} counts)",
        f"  near practical high saturation taxels (baseline mean >= {summary['sensor']['practical_high_saturation_threshold_counts']} counts): {b['near_practical_high_saturation_taxels']}",
        f"  rejected packets / counter skips: CRC={b['parser_stats']['crc_errors']}, format={b['parser_stats']['format_errors']}, counter_skips={b['parser_stats']['counter_skips']}",
        f"  serial timeouts: {b['parser_stats']['timeouts']}",
        f"  host valid frames dropped: {b['parser_stats']['host_discarded_valid_frames']}",
        f"  pre-capture / settling valid frames discarded: {b['parser_stats']['pre_capture_frames_discarded']} / {b['parser_stats']['settling_frames_discarded']}",
        f"  receiver finalization after cutoff: {b['parser_stats']['capture_stop_lag_s']:.6f} s",
        f"  acquisition batches: {b['parser_stats']['acquisition_batches']}, maximum frames/batch: {b['parser_stats']['maximum_frames_in_batch']}, maximum RX buffer: {b['parser_stats']['maximum_rxbuf_bytes']} bytes",
    ]

    if "touch" in summary:
        t = summary["touch"]
        lines += [
            "",
            "Touch/contact signal:",
            f"  requested duration: {t['parser_stats']['requested_duration_s']:.3f} s",
            f"  captured frames: {t['frames']} over {t['duration_s']:.3f} s",
            f"  valid frame rate: {t['fps_hz']:.2f} Hz",
            f"  STM32 counter span (inclusive): {t['parser_stats']['counter_span_frames']} frame attempts",
            f"  valid frame fraction: {t['parser_stats']['valid_frame_fraction']:.6f} ({t['parser_stats']['valid_frame_percent']:.3f}%)",
            f"  counter range: {t['parser_stats']['first_counter']} -> {t['parser_stats']['last_counter']}",
            f"  counter discontinuities: {t['parser_stats']['counter_discontinuities']}, largest step: {t['parser_stats']['largest_counter_jump']}, largest missing gap: {t['parser_stats']['largest_missing_counter_gap']}",
            f"  |signal| median / max: {t['abs_delta_counts']['median']:.2f} / {t['abs_delta_counts']['max']:.2f} ADC counts",
            f"  SNR median / max: {t['snr']['median']:.2f} / {t['snr']['max']:.2f}",
            f"  reliable taxels: {t['reliable_taxels']} / {summary['sensor']['taxels']}",
            f"  reliability rule: |signal| >= {t['reliability_rule']['min_signal_counts']:.1f} counts and SNR >= {t['reliability_rule']['snr_threshold']:.1f}",
            f"  rejected packets / counter skips: CRC={t['parser_stats']['crc_errors']}, format={t['parser_stats']['format_errors']}, counter_skips={t['parser_stats']['counter_skips']}",
            f"  serial timeouts: {t['parser_stats']['timeouts']}",
            f"  host valid frames dropped: {t['parser_stats']['host_discarded_valid_frames']}",
            f"  pre-capture / settling valid frames discarded: {t['parser_stats']['pre_capture_frames_discarded']} / {t['parser_stats']['settling_frames_discarded']}",
            f"  receiver finalization after cutoff: {t['parser_stats']['capture_stop_lag_s']:.6f} s",
            f"  acquisition batches: {t['parser_stats']['acquisition_batches']}, maximum frames/batch: {t['parser_stats']['maximum_frames_in_batch']}, maximum RX buffer: {t['parser_stats']['maximum_rxbuf_bytes']} bytes",
        ]

    lines += [
        "",
        "Thesis note:",
        f"  Acquisition interval: {b['parser_stats']['capture_boundary_policy']}. Host monotonic boundaries do not provide firmware sample timestamps.",
        "  The STM32 frame counter tracks calls to Tactile_SendFrameUSB() (after successful scans); it cannot detect failed scans.",
        "  Use the no-touch baseline for temporal noise and drift. Use the touch/contact part only as a signal demonstration unless the applied force is controlled.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure tactile sensor noise with packed12+CRC serial packets.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/cu.usbmodem2078388A45461")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--header_size", type=int, default=DEFAULT_HEADER_SIZE)
    parser.add_argument("--crc_size", type=int, default=DEFAULT_CRC_SIZE, choices=[0, 2])
    parser.add_argument("--payload_format", choices=["packed12", "uint16"], default="packed12")
    parser.add_argument("--order", choices=["col-major", "row-major"], default="col-major")
    parser.add_argument("--no_flip_lr", action="store_true", help="Disable left/right flip")
    parser.add_argument("--adc_max", type=int, default=DEFAULT_ADC_MAX)
    parser.add_argument(
        "--practical_high_saturation",
        type=int,
        default=DEFAULT_PRACTICAL_HIGH_SATURATION,
        help="Baseline mean >= this ADC count is near practical analog high saturation; separate from adc_max",
    )
    parser.add_argument("--baseline_sec", type=float, default=60.0)
    parser.add_argument("--touch_sec", type=float, default=10.0)
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--warmup_timeout_sec", type=float, default=10.0,
                        help="Fail if valid startup frames are not received within this interval")
    parser.add_argument("--settle_sec", type=float, default=0.5)
    parser.add_argument("--serial_timeout", type=float, default=0.02)
    parser.add_argument("--min_signal_counts", type=float, default=20.0)
    parser.add_argument("--snr_threshold", type=float, default=5.0)
    parser.add_argument("--saturation_margin", type=int, default=20)
    parser.add_argument("--skip_touch", action="store_true")
    parser.add_argument("--trial_name", default="noise_test")
    parser.add_argument("--notes", default="")
    parser.add_argument("--out_root", default=".")

    # Fixed heatmap scales for thesis figures.
    parser.add_argument("--raw_vmin", type=float, default=1900.0)
    parser.add_argument("--raw_vmax", type=float, default=4095.0)
    parser.add_argument("--baseline_zoom_vmin", type=float, default=1900.0)
    parser.add_argument("--baseline_zoom_vmax", type=float, default=2100.0)
    parser.add_argument("--signal_vmin", type=float, default=0.0)
    parser.add_argument("--signal_vmax", type=float, default=2200.0)
    parser.add_argument("--noise_std_vmin", type=float, default=0.0)
    parser.add_argument("--noise_std_vmax", type=float, default=20.0)
    parser.add_argument("--noise_p2p_vmin", type=float, default=0.0)
    parser.add_argument("--noise_p2p_vmax", type=float, default=80.0)
    parser.add_argument("--drift_vmin", type=float, default=0.0)
    parser.add_argument("--drift_vmax", type=float, default=20.0)
    parser.add_argument("--snr_vmin", type=float, default=0.0)
    parser.add_argument("--snr_vmax", type=float, default=20.0)
    parser.add_argument("--save_dynamic_debug", action="store_true", help="Also save dynamic min/max debug heatmaps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_trial_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.trial_name)
    out_dir = Path(args.out_root) / f"{safe_trial_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    print(f"[INFO] Opening {args.port} at {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=args.serial_timeout)
    receiver = None
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        reader = TactilePacketReader(
            ser=ser,
            rows=args.rows,
            cols=args.cols,
            header_size=args.header_size,
            crc_size=args.crc_size,
            payload_format=args.payload_format,
            order=args.order,
            flip_lr=not args.no_flip_lr,
            adc_max=args.adc_max,
        )

        receiver = ContinuousReceiver(reader)
        receiver.start()

        print("[INFO] Packet setup:")
        print(f"       values={reader.num_values}, payload={reader.payload_size} bytes, packet={reader.packet_size} bytes")
        print(f"       orientation={'viewer-compatible fliplr(rows x cols)' if not args.no_flip_lr else 'rows x cols, no flip'}")

        print("[INFO] Discarding startup/warmup frames...")
        warmup_stats = discard_fresh_frames(receiver, args.warmup_frames, args.warmup_timeout_sec)
        if any(v for k, v in warmup_stats.items() if k != "valid_frames_discarded"):
            print(f"[INFO] Warmup parser stats: {warmup_stats}")

        print("[INFO] For final stability measurements, allow the sensor/electronics")
        print("       to thermally/electrically stabilize before starting the recording.")
        input("\nDo NOT touch the sensor. Press ENTER to record the no-touch baseline/noise...")
        # The receiver keeps draining DISCARD batches during input and settling.
        settling_before = receiver.snapshot_stats()["valid_frames_discarded"]
        if args.settle_sec > 0:
            time.sleep(args.settle_sec)
        settling_discarded = receiver.snapshot_stats()["valid_frames_discarded"] - settling_before
        baseline_frames, baseline_fps, baseline_stats = capture_frames(
            receiver, args.baseline_sec, settling_discarded,
        )
        baseline = compute_capture_matrices(baseline_frames)

        high_sat = baseline["mean"] > (args.adc_max - args.saturation_margin)
        low_sat = baseline["mean"] < args.saturation_margin
        practical_high_sat = baseline["mean"] >= args.practical_high_saturation

        print("\n========== NO-TOUCH BASELINE / NOISE ==========")
        print_capture_summary(baseline_frames, baseline_fps, baseline_stats)
        print(f"Noise std median: {np.median(baseline['std']):.2f} counts")
        print(f"Noise std max:    {np.max(baseline['std']):.2f} counts")
        print(f"Noise p2p median: {np.median(baseline['p2p']):.2f} counts")
        print(f"Noise p2p max:    {np.max(baseline['p2p']):.2f} counts")
        print(f"Abs. drift median: {np.median(baseline['abs_drift']):.2f} counts")
        print(f"Abs. drift max:    {np.max(baseline['abs_drift']):.2f} counts")
        print(f"Near ADC high saturation, baseline mean > {args.adc_max - args.saturation_margin}: {int(high_sat.sum())} taxels")
        print(f"Near ADC low saturation, baseline mean < {args.saturation_margin}: {int(low_sat.sum())} taxels")
        print(f"Near practical high saturation, baseline mean >= {args.practical_high_saturation}: {int(practical_high_sat.sum())} taxels")
        if baseline_stats["counter_skips"] or baseline_stats["crc_errors"] or baseline_stats["format_errors"]:
            print("[WARN] Incomplete/corrupt stream: retain this run for debugging, "
                  "but do not report it as loss-free thesis data.")

        touch_frames = None
        touch = None
        touch_fps = None
        touch_stats = None

        if not args.skip_touch:
            input("\nNow apply/hold the object or cloth contact. Press ENTER to record touch/contact signal...")
            settling_before = receiver.snapshot_stats()["valid_frames_discarded"]
            if args.settle_sec > 0:
                time.sleep(args.settle_sec)
            settling_discarded = receiver.snapshot_stats()["valid_frames_discarded"] - settling_before
            touch_frames, touch_fps, touch_stats = capture_frames(
                receiver, args.touch_sec, settling_discarded,
            )
            touch_base = compute_capture_matrices(touch_frames)
            delta = touch_base["mean"] - baseline["mean"]
            abs_delta = np.abs(delta)
            snr = abs_delta / (baseline["std"] + 1e-6)
            reliable = (abs_delta >= args.min_signal_counts) & (snr >= args.snr_threshold)
            touch = {**touch_base, "delta": delta, "abs_delta": abs_delta, "snr": snr, "reliable": reliable}

            print("\n========== TOUCH / CONTACT SIGNAL ==========")
            print_capture_summary(touch_frames, touch_fps, touch_stats)
            print(f"Signal |delta| median: {np.median(abs_delta):.2f} counts")
            print(f"Signal |delta| max:    {np.max(abs_delta):.2f} counts")
            print(f"SNR median:            {np.median(snr):.2f}")
            print(f"SNR max:               {np.max(snr):.2f}")
            print(f"Reliable taxels:       {int(reliable.sum())} / {args.rows * args.cols}")
            print("Rule used:")
            print(f"  reliable if |signal| >= {args.min_signal_counts:.1f} counts")
            print(f"  and |signal|/noise_std >= {args.snr_threshold:.1f}")

        npz_path = out_dir / "noise_results.npz"
        # Omit absent touch arrays. Saving None produces pickle-backed object
        # arrays, which break safe np.load(..., allow_pickle=False) later.
        archive = {
            "baseline_frames": baseline_frames,
            "baseline_counters": baseline_stats["counters"],
            "baseline_mean": baseline["mean"],
            "baseline_median": baseline["median"],
            "noise_std": baseline["std"],
            "noise_p2p": baseline["p2p"],
            "baseline_drift": baseline["drift_second_half_minus_first_half"],
            "baseline_abs_drift": baseline["abs_drift"],
        }
        if touch is not None and touch_stats is not None:
            archive.update({
                "touch_frames": touch_frames,
                "touch_counters": touch_stats["counters"],
                "touch_mean": touch["mean"],
                "delta": touch["delta"],
                "abs_delta": touch["abs_delta"],
                "snr": touch["snr"],
                "reliable": touch["reliable"],
            })
        np.savez_compressed(npz_path, **archive)

        summary = {
            "trial_name": args.trial_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "notes": args.notes,
            "script": Path(__file__).name,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sensor": {
                "rows": args.rows,
                "cols": args.cols,
                "taxels": args.rows * args.cols,
                "adc_max": args.adc_max,
                "saturation_margin_counts": args.saturation_margin,
                "practical_high_saturation_threshold_counts": args.practical_high_saturation,
                "saturation_basis": "baseline mean; near_high_saturation uses > adc_max - saturation_margin_counts; near_practical_high_saturation uses >= practical_high_saturation_threshold_counts",
            },
            "packet": {
                "sync_hex": SYNC.hex(" "),
                "header_size_bytes": args.header_size,
                "payload_format": args.payload_format,
                "payload_size_bytes": reader.payload_size,
                "crc_size_bytes": args.crc_size,
                "packet_size_bytes": reader.packet_size,
                "crc_algorithm": "CRC16-CCITT-FALSE over frame_counter + payload" if args.crc_size else "none",
                "order": args.order,
                "flip_lr_to_match_live_viewer": not args.no_flip_lr,
            },
            "run_arguments": vars(args),
            "warmup": warmup_stats,
            "receiver_lifetime_stats": receiver.snapshot_stats(),
            "baseline": {
                "frames": int(len(baseline_frames)),
                "duration_s": float(baseline_stats["duration_s"]),
                "fps_hz": float(baseline_fps),
                "parser_stats": json_safe(baseline_stats),
                "baseline_mean_counts": matrix_summary(baseline["mean"]),
                "noise_std_counts": matrix_summary(baseline["std"]),
                "noise_p2p_counts": matrix_summary(baseline["p2p"]),
                "drift_counts": matrix_summary(baseline["drift_second_half_minus_first_half"]),
                "abs_drift_counts": matrix_summary(baseline["abs_drift"]),
                "near_low_saturation_taxels": int(low_sat.sum()),
                "near_high_saturation_taxels": int(high_sat.sum()),
                "near_practical_high_saturation_taxels": int(practical_high_sat.sum()),
            },
        }

        if touch is not None and touch_stats is not None:
            summary["touch"] = {
                "frames": int(len(touch_frames)),
                "duration_s": float(touch_stats["duration_s"]),
                "fps_hz": float(touch_fps),
                "parser_stats": json_safe(touch_stats),
                "touch_mean_counts": matrix_summary(touch["mean"]),
                "delta_counts": matrix_summary(touch["delta"]),
                "abs_delta_counts": matrix_summary(touch["abs_delta"]),
                "snr": matrix_summary(touch["snr"]),
                "reliable_taxels": int(touch["reliable"].sum()),
                "reliability_rule": {
                    "min_signal_counts": float(args.min_signal_counts),
                    "snr_threshold": float(args.snr_threshold),
                },
            }

        summary_json_path = out_dir / "run_summary.json"
        with open(summary_json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)

        thesis_txt_path = out_dir / "thesis_summary.txt"
        thesis_txt_path.write_text(make_thesis_summary_text(summary), encoding="utf-8")

        summary_csv_path = out_dir / "thesis_summary.csv"
        save_summary_csv(summary_csv_path, summary)

        per_taxel_csv_path = out_dir / "per_taxel_results.csv"
        save_per_taxel_csv(
            per_taxel_csv_path,
            args.rows,
            args.cols,
            baseline,
            adc_max=args.adc_max,
            saturation_margin=args.saturation_margin,
            touch=touch,
            practical_high_saturation=args.practical_high_saturation,
        )

        # Fixed-scale thesis heatmaps.
        save_heatmap_png(out_dir / "baseline_mean_raw_fixed.png", baseline["mean"], "Baseline mean ADC, fixed scale", "ADC counts", vmin=args.raw_vmin, vmax=args.raw_vmax)
        save_heatmap_png(out_dir / "baseline_mean_zoom_fixed.png", baseline["mean"], "Baseline mean ADC, zoomed fixed scale", "ADC counts", vmin=args.baseline_zoom_vmin, vmax=args.baseline_zoom_vmax)
        save_heatmap_png(out_dir / "noise_std_fixed.png", baseline["std"], "No-touch noise standard deviation, fixed scale", "ADC counts", vmin=args.noise_std_vmin, vmax=args.noise_std_vmax)
        save_heatmap_png(out_dir / "noise_peak_to_peak_fixed.png", baseline["p2p"], "No-touch noise peak-to-peak, fixed scale", "ADC counts", vmin=args.noise_p2p_vmin, vmax=args.noise_p2p_vmax)
        save_heatmap_png(out_dir / "baseline_abs_drift_fixed.png", baseline["abs_drift"], "Absolute baseline drift, fixed scale", "ADC counts", vmin=args.drift_vmin, vmax=args.drift_vmax)
        save_heatmap_png(out_dir / "saturation_mask.png", high_sat.astype(float) - low_sat.astype(float), "Saturation mask: high=1, low=-1", "mask", vmin=-1, vmax=1)

        if touch is not None:
            save_heatmap_png(out_dir / "touch_mean_raw_fixed.png", touch["mean"], "Touch/contact mean ADC, fixed scale", "ADC counts", vmin=args.raw_vmin, vmax=args.raw_vmax)
            save_heatmap_png(out_dir / "signed_signal_delta_fixed.png", touch["delta"], "Signed signal: touch - baseline, fixed scale", "ADC counts", vmin=-args.signal_vmax, vmax=args.signal_vmax)
            save_heatmap_png(out_dir / "abs_signal_fixed.png", touch["abs_delta"], "Absolute touch/contact signal, fixed scale", "ADC counts", vmin=args.signal_vmin, vmax=args.signal_vmax)
            save_heatmap_png(out_dir / "snr_fixed.png", touch["snr"], "Signal-to-noise ratio, fixed scale", "SNR", vmin=args.snr_vmin, vmax=args.snr_vmax)
            save_heatmap_png(out_dir / "reliable_taxels.png", touch["reliable"].astype(float), "Reliable taxels", "0/1", vmin=0, vmax=1)

        # Optional dynamic heatmaps for debugging only.
        if args.save_dynamic_debug:
            save_heatmap_png(out_dir / "baseline_mean_dynamic_debug.png", baseline["mean"], "Baseline mean ADC, dynamic debug", "ADC counts")
            save_heatmap_png(out_dir / "noise_std_dynamic_debug.png", baseline["std"], "No-touch noise std, dynamic debug", "ADC counts")
            save_heatmap_png(out_dir / "noise_peak_to_peak_dynamic_debug.png", baseline["p2p"], "No-touch noise p2p, dynamic debug", "ADC counts")
            if touch is not None:
                save_heatmap_png(out_dir / "touch_mean_dynamic_debug.png", touch["mean"], "Touch/contact mean ADC, dynamic debug", "ADC counts")
                save_heatmap_png(out_dir / "abs_signal_dynamic_debug.png", touch["abs_delta"], "Absolute signal, dynamic debug", "ADC counts")
                save_heatmap_png(out_dir / "snr_dynamic_debug.png", touch["snr"], "SNR, dynamic debug", "SNR")

        print("\n========== SAVED ==========")
        print(f"Folder:       {out_dir}")
        print(f"Summary JSON: {summary_json_path}")
        print(f"Thesis TXT:   {thesis_txt_path}")
        print(f"Summary CSV:  {summary_csv_path}")
        print(f"Per-taxel CSV:{per_taxel_csv_path}")
        print(f"Raw NPZ:      {npz_path}")

    finally:
        if receiver is not None:
            receiver.stop()
        ser.close()


if __name__ == "__main__":
    main()
