import argparse
import csv
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial


MAGIC = b"\xAA\x55"


class BadFrameError(Exception):
    """Raised when a packet is complete but cannot be valid ADC data."""


def read_exact(ser, n):
    data = bytearray()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def read_frame(ser, rows, cols, header_bytes=6, order="col-major", adc_max=4095):
    """
    Reads one valid binary tactile frame.

    Expected format:
      MAGIC: 2 bytes = AA 55
      header: total header_bytes bytes, including magic
              for the current firmware this is AA 55 + uint32 frame_counter
      payload: rows * cols uint16 values, little endian

    If your STM32 stores frame as tactile_frame[COLS][ROWS],
    use order='col-major' so output becomes (ROWS, COLS).

    Important: this function rejects packets with values outside the ADC range.
    That protects the noise statistics from false AA55 sync hits inside payload data.
    """

    if header_bytes < 2:
        raise ValueError("header_bytes must include the 2 magic bytes, so it must be >= 2")

    # Find magic bytes AA 55.
    window = b""
    while True:
        b = ser.read(1)
        if not b:
            raise TimeoutError("Timeout while searching for frame magic bytes.")
        window = (window + b)[-2:]
        if window == MAGIC:
            break

    # Read rest of header.
    rest_header = header_bytes - 2
    h = b""
    if rest_header > 0:
        h = read_exact(ser, rest_header)
        if h is None:
            raise TimeoutError("Timeout while reading frame header.")

    # Current firmware header is AA55 + uint32 frame counter.
    frame_counter = None
    if rest_header >= 4:
        frame_counter = struct.unpack("<I", h[:4])[0]

    payload_bytes = rows * cols * 2
    payload = read_exact(ser, payload_bytes)
    if payload is None:
        raise TimeoutError("Timeout while reading frame payload.")

    data = np.frombuffer(payload, dtype="<u2")  # little-endian uint16

    if data.size != rows * cols:
        raise BadFrameError(f"Wrong payload size: got {data.size}, expected {rows * cols}")

    # A real 12-bit ADC sample can never be above 4095.
    # If this happens, the parser almost certainly synced to AA55 inside payload bytes.
    dmax = int(data.max()) if data.size else -1
    if dmax > int(adc_max):
        raise BadFrameError(f"ADC value outside valid range: max={dmax} > adc_max={adc_max}")

    if order == "col-major":
        # Firmware buffer often: tactile_frame[COLS][ROWS]
        frame = data.reshape(cols, rows).T
    elif order == "row-major":
        # Firmware buffer: tactile_frame[ROWS][COLS]
        frame = data.reshape(rows, cols)
    else:
        raise ValueError("order must be 'col-major' or 'row-major'")

    return frame.astype(np.float32), frame_counter


def discard_fresh_frames(ser, n, rows, cols, header_bytes, order, adc_max):
    """Discard a few valid frames after flushing so capture starts on fresh data."""
    bad = 0
    timeouts = 0
    for _ in range(n):
        try:
            read_frame(
                ser,
                rows=rows,
                cols=cols,
                header_bytes=header_bytes,
                order=order,
                adc_max=adc_max,
            )
        except TimeoutError:
            timeouts += 1
        except BadFrameError:
            bad += 1
    return bad, timeouts


def capture_frames(ser, duration_s, rows, cols, header_bytes, order, adc_max):
    frames = []
    t0 = time.perf_counter()
    timeout_count = 0
    bad_frame_count = 0
    counter_skip_count = 0
    last_counter = None

    while time.perf_counter() - t0 < duration_s:
        try:
            frame, counter = read_frame(
                ser,
                rows=rows,
                cols=cols,
                header_bytes=header_bytes,
                order=order,
                adc_max=adc_max,
            )

            if counter is not None and last_counter is not None:
                expected = (last_counter + 1) & 0xFFFFFFFF
                if counter != expected:
                    counter_skip_count += (counter - expected) & 0xFFFFFFFF
            if counter is not None:
                last_counter = counter

            frames.append(frame)

        except TimeoutError:
            timeout_count += 1
        except BadFrameError:
            bad_frame_count += 1
            # Do not use this packet for statistics. Next iteration will search for AA55 again.
            continue

    if len(frames) == 0:
        raise RuntimeError(
            "No valid frames captured. Check port, frame format, header size, and ADC range."
        )

    frames = np.stack(frames, axis=0)  # (N, rows, cols)
    actual_duration = time.perf_counter() - t0
    fps = len(frames) / actual_duration

    stats = {
        "timeouts": timeout_count,
        "bad_frames_rejected": bad_frame_count,
        "counter_skips": counter_skip_count,
    }

    return frames, fps, stats


def save_heatmap_png(path, data, title):
    """Save a heatmap. Uses matplotlib when available, otherwise falls back to OpenCV."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.imshow(data, aspect="auto")
        plt.colorbar(label="ADC counts")
        plt.title(title)
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return
    except Exception:
        pass

    try:
        import cv2

        arr = np.asarray(data, dtype=np.float32)
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        if mx > mn:
            norm = (arr - mn) * 255.0 / (mx - mn)
        else:
            norm = np.zeros_like(arr)
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
        img = cv2.resize(img, (960, 360), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(path), img)
    except Exception as e:
        print(f"[WARN] Could not save heatmap {path.name}: {e}")


def print_capture_summary(name, frames, fps, stats):
    print(f"Frames captured: {len(frames)}")
    print(f"Approx. valid frame rate: {fps:.2f} Hz")
    print(f"Timeouts: {stats['timeouts']}")
    print(f"Bad/corrupt frames rejected: {stats['bad_frames_rejected']}")
    print(f"STM32 counter skips: {stats['counter_skips']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/cu.usbmodem2078388A45461")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--header_bytes", type=int, default=6, help="Total header size including AA55 magic")
    parser.add_argument("--order", choices=["col-major", "row-major"], default="col-major")
    parser.add_argument("--baseline_sec", type=float, default=10.0)
    parser.add_argument("--touch_sec", type=float, default=5.0)
    parser.add_argument("--warmup_frames", type=int, default=20)
    parser.add_argument("--min_signal_counts", type=float, default=20.0)
    parser.add_argument("--snr_threshold", type=float, default=5.0)
    parser.add_argument("--adc_max", type=int, default=4095)
    parser.add_argument("--skip_touch", action="store_true", help="Only measure no-touch noise")
    args = parser.parse_args()

    out_dir = Path(f"noise_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(exist_ok=True)

    print(f"[INFO] Opening {args.port} at {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=0.25)
    time.sleep(0.2)

    # Very important: remove old queued data from before the script started.
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("[INFO] Discarding warmup frames...")
    warmup_bad, warmup_timeouts = discard_fresh_frames(
        ser,
        args.warmup_frames,
        args.rows,
        args.cols,
        args.header_bytes,
        args.order,
        args.adc_max,
    )
    if warmup_bad or warmup_timeouts:
        print(f"[INFO] Warmup rejected bad={warmup_bad}, timeouts={warmup_timeouts}")

    input("\nDo NOT touch the sensor. Press ENTER to record baseline/noise...")

    # Very important: while you were waiting at the ENTER prompt, the serial buffer filled up.
    # Flush it so the measured FPS and noise come from live frames, not old queued frames.
    ser.reset_input_buffer()
    discard_fresh_frames(
        ser,
        max(3, min(args.warmup_frames, 10)),
        args.rows,
        args.cols,
        args.header_bytes,
        args.order,
        args.adc_max,
    )

    baseline_frames, baseline_fps, baseline_stats = capture_frames(
        ser,
        duration_s=args.baseline_sec,
        rows=args.rows,
        cols=args.cols,
        header_bytes=args.header_bytes,
        order=args.order,
        adc_max=args.adc_max,
    )

    baseline_mean = baseline_frames.mean(axis=0)
    noise_std = baseline_frames.std(axis=0)
    noise_p2p = baseline_frames.max(axis=0) - baseline_frames.min(axis=0)

    print("\n========== NO-TOUCH NOISE ==========")
    print_capture_summary("baseline", baseline_frames, baseline_fps, baseline_stats)
    print(f"Noise std median: {np.median(noise_std):.2f} counts")
    print(f"Noise std max:    {np.max(noise_std):.2f} counts")
    print(f"Noise p2p median: {np.median(noise_p2p):.2f} counts")
    print(f"Noise p2p max:    {np.max(noise_p2p):.2f} counts")

    high_sat = baseline_mean > (args.adc_max - 20)
    low_sat = baseline_mean < 20

    print(f"Near high saturation > {args.adc_max - 20}: {int(high_sat.sum())} taxels")
    print(f"Near low saturation  < 20: {int(low_sat.sum())} taxels")

    touch_frames = None
    touch_mean = None
    touch_stats = None
    touch_fps = None
    delta = None
    abs_delta = None
    snr = None
    reliable = None

    if not args.skip_touch:
        input("\nNow touch/hold the cloth on the sensor. Press ENTER to record touch signal...")

        # Again flush old queued frames from the time spent at the prompt.
        ser.reset_input_buffer()
        discard_fresh_frames(
            ser,
            max(3, min(args.warmup_frames, 10)),
            args.rows,
            args.cols,
            args.header_bytes,
            args.order,
            args.adc_max,
        )

        touch_frames, touch_fps, touch_stats = capture_frames(
            ser,
            duration_s=args.touch_sec,
            rows=args.rows,
            cols=args.cols,
            header_bytes=args.header_bytes,
            order=args.order,
            adc_max=args.adc_max,
        )

        touch_mean = touch_frames.mean(axis=0)
        delta = touch_mean - baseline_mean
        abs_delta = np.abs(delta)

        snr = abs_delta / (noise_std + 1e-6)

        reliable = (
            (abs_delta >= args.min_signal_counts)
            & (snr >= args.snr_threshold)
        )

        print("\n========== TOUCH / CLOTH SIGNAL ==========")
        print_capture_summary("touch", touch_frames, touch_fps, touch_stats)
        print(f"Signal median: {np.median(abs_delta):.2f} counts")
        print(f"Signal max:    {np.max(abs_delta):.2f} counts")
        print(f"SNR median:    {np.median(snr):.2f}")
        print(f"SNR max:       {np.max(snr):.2f}")
        print(f"Reliable taxels: {int(reliable.sum())} / {args.rows * args.cols}")

        print("\nRule used:")
        print(f"Reliable if signal >= {args.min_signal_counts:.1f} counts")
        print(f"and signal/noise >= {args.snr_threshold:.1f}")

    # Save raw data
    npz_path = out_dir / "noise_results.npz"
    np.savez(
        npz_path,
        baseline_frames=baseline_frames,
        baseline_mean=baseline_mean,
        noise_std=noise_std,
        noise_p2p=noise_p2p,
        baseline_fps=baseline_fps,
        baseline_timeouts=baseline_stats["timeouts"],
        baseline_bad_frames_rejected=baseline_stats["bad_frames_rejected"],
        baseline_counter_skips=baseline_stats["counter_skips"],
        touch_frames=touch_frames,
        touch_mean=touch_mean,
        touch_fps=touch_fps,
        touch_timeouts=None if touch_stats is None else touch_stats["timeouts"],
        touch_bad_frames_rejected=None if touch_stats is None else touch_stats["bad_frames_rejected"],
        touch_counter_skips=None if touch_stats is None else touch_stats["counter_skips"],
        delta=delta,
        abs_delta=abs_delta,
        snr=snr,
        reliable=reliable,
    )

    # Save CSV per taxel
    csv_path = out_dir / "per_taxel_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = [
            "row",
            "col",
            "baseline_mean",
            "noise_std",
            "noise_peak_to_peak",
        ]

        if not args.skip_touch:
            header += [
                "touch_mean",
                "delta_touch_minus_baseline",
                "abs_delta",
                "snr",
                "reliable",
            ]

        writer.writerow(header)

        for r in range(args.rows):
            for c in range(args.cols):
                row = [
                    r,
                    c,
                    float(baseline_mean[r, c]),
                    float(noise_std[r, c]),
                    float(noise_p2p[r, c]),
                ]

                if not args.skip_touch:
                    row += [
                        float(touch_mean[r, c]),
                        float(delta[r, c]),
                        float(abs_delta[r, c]),
                        float(snr[r, c]),
                        int(reliable[r, c]),
                    ]

                writer.writerow(row)

    # Save heatmaps
    save_heatmap_png(out_dir / "baseline_mean.png", baseline_mean, "Baseline mean ADC")
    save_heatmap_png(out_dir / "noise_std.png", noise_std, "No-touch noise std")
    save_heatmap_png(out_dir / "noise_peak_to_peak.png", noise_p2p, "No-touch noise peak-to-peak")

    if not args.skip_touch:
        save_heatmap_png(out_dir / "abs_signal.png", abs_delta, "Absolute touch signal")
        save_heatmap_png(out_dir / "snr.png", snr, "Signal-to-noise ratio")

    print("\n========== SAVED ==========")
    print(f"Folder: {out_dir}")
    print(f"CSV:    {csv_path}")
    print(f"NPZ:    {npz_path}")


if __name__ == "__main__":
    main()
