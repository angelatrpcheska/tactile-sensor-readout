"""Live viewer for legacy 100 Hz STM32 uint16 ADC frames without CRC.

Run from the repository root:
    python legacy/read_tactile_threaded_100Hz.py --port /dev/ttyACM0
"""

import argparse
import struct
import serial
import numpy as np
import cv2
import time
import threading
from pathlib import Path

NUM_COLS = 32
NUM_ROWS = 12

SYNC = b"\xAA\x55"
HEADER_SIZE = 6
DATA_SIZE = NUM_COLS * NUM_ROWS * 2
PACKET_SIZE = HEADER_SIZE + DATA_SIZE

def read_exact(ser, n):
    data = bytearray()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def find_sync(ser):
    prev = b""

    while True:
        b = ser.read(1)
        if not b:
            return False

        if prev + b == SYNC:
            return True

        prev = b

def read_frame(ser):
    """Read one frame for debugging; the live viewer uses the buffered reader."""
    if not find_sync(ser):
        return None, None

    counter_bytes = read_exact(ser, 4)
    if counter_bytes is None:
        return None, None

    frame_counter = struct.unpack("<I", counter_bytes)[0]

    payload = read_exact(ser, DATA_SIZE)
    if payload is None:
        return None, None

    values = np.frombuffer(payload, dtype="<u2")  # little-endian uint16

    if values is None:
         return None, None

    matrix = values.reshape((NUM_COLS, NUM_ROWS)).copy()

    return frame_counter, matrix


def read_latest_available_frame(ser, rxbuf):
    """Parse buffered packets and return the newest frame and valid-frame count.

    Older complete frames are discarded to limit display latency. Incomplete
    packets remain buffered for the next read.
    """
    # An empty-buffer read uses the serial timeout so the thread can stop promptly.
    n = ser.in_waiting

    if n > 0:
        chunk = ser.read(n)
    else:
        chunk = ser.read(1)

    if not chunk:
        return None, None, 0

    rxbuf.extend(chunk)

    latest_counter = None
    latest_matrix = None
    frames_parsed = 0

    while True:
        sync_idx = rxbuf.find(SYNC)

        if sync_idx < 0:
            # Retain a possible first sync byte spanning serial reads.
            if len(rxbuf) > 1:
                rxbuf[:] = rxbuf[-1:]
            break

        if sync_idx > 0:
            del rxbuf[:sync_idx]

        if len(rxbuf) < PACKET_SIZE:
            break

        # Layout: sync [0:2], little-endian counter [2:6], uint16 ADCs [6:774].
        # This firmware profile has no CRC; packed12+CRC is handled by the current viewer.
        packet = bytes(rxbuf[:PACKET_SIZE])
        del rxbuf[:PACKET_SIZE]
        counter = struct.unpack("<I", packet[2:6])[0]
        payload = packet[6:]

        values = np.frombuffer(payload, dtype="<u2")
        if values.size != NUM_COLS * NUM_ROWS:
            continue

        if values is None:
            continue

        latest_counter = counter
        latest_matrix = values.reshape((NUM_COLS, NUM_ROWS)).copy()
        frames_parsed += 1

    return latest_counter, latest_matrix, frames_parsed


def serial_reader_thread(ser, state, lock, stop_event):
    """Drain serial input in the background, retaining the newest frame.

    Track parsed frames separately from frames omitted for display and counter
    gaps, so deliberate display drops are not reported as missing sensor frames.
    """
    rxbuf = bytearray()
    last_received_counter = None

    while not stop_event.is_set():
        counter, matrix, parsed_count = read_latest_available_frame(ser, rxbuf)

        if matrix is None:
            continue

        stm32_skips = 0
        if last_received_counter is not None:
            # The newest counter advances by parsed_count, modulo uint32.
            expected = (last_received_counter + parsed_count) & 0xFFFFFFFF

            if counter != expected:
                # Unaccounted counter gaps can include firmware drops under USB load.
                stm32_skips = (counter - expected) & 0xFFFFFFFF

        last_received_counter = counter

        with lock:
            # Replace the previous frame rather than queue stale display updates.
            state["counter"] = counter
            state["matrix"] = matrix
            state["received_frames"] += parsed_count
            state["reader_dropped_frames"] += max(0, parsed_count - 1)
            state["stm32_counter_skips"] += stm32_skips


def matrix_to_image(
    matrix,
    dynamic=False,
    counter=None,
    display_fps=0.0,
    received_fps=0.0,
    show_values=False,
    show_header=False,
    baseline_matrix=None,
    raw_vmin=1900.0,
    raw_vmax=4050.0,
    signal_vmax=2100.0,
    display_threshold=250.0,
):
    """Render raw ADC values or a baseline-subtracted, display-filtered heatmap."""

    # Convert STM32 [column][row] (32x12) to display (12x32), then mirror columns.
    raw_display = np.fliplr(matrix.T).astype(np.float32)

    if baseline_matrix is not None:
        baseline_display = np.fliplr(
            baseline_matrix.T
        ).astype(np.float32)

        # Preserve signed float differences for statistics and labels; threshold colours only.
        display = raw_display - baseline_display

        # Contact is expected to increase the ADC value.
        positive_signal = np.clip(display, 0.0, None)

        # Suppress weak responses only in the visualized heatmap.
        color_values = np.where(
            positive_signal >= display_threshold,
            positive_signal,
            0.0,
        )

        fixed_vmin = 0.0
        fixed_vmax = signal_vmax
        display_mode = "delta"
    else:
        display = raw_display
        color_values = raw_display
        fixed_vmin = raw_vmin
        fixed_vmax = raw_vmax
        display_mode = "raw"

    rows, cols = display.shape

    # Limit cell size to reduce OpenCV drawing latency.
    cell_w = 30
    cell_h = 25
    header_h = 40

    grid_w = cols * cell_w
    grid_h = rows * cell_h

    if dynamic:
        mn = float(color_values.min())
        mx = float(color_values.max())
        if mx > mn:
            norm = (color_values - mn) * 255.0 / (mx - mn)
        else:
            norm = np.zeros_like(color_values)
    else:
        if fixed_vmax <= fixed_vmin:
            raise ValueError("Display maximum must be larger than display minimum.")

        norm = (
            (color_values - fixed_vmin)
            * 255.0
            / (fixed_vmax - fixed_vmin)
        )

    norm = np.clip(norm, 0, 255).astype(np.uint8)

    color_small = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    grid_img = cv2.resize(
        color_small,
        (grid_w, grid_h),
        interpolation=cv2.INTER_NEAREST
    )

    if show_header:
        img = np.zeros((header_h + grid_h, grid_w, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        img[header_h:header_h + grid_h, 0:grid_w] = grid_img

        if counter is None:
            title = (
                f"mode={display_mode}  recv={received_fps:.1f} Hz  "
                f"disp={display_fps:.1f} Hz  min={int(display.min())}  "
                f"max={int(display.max())}  mean={display.mean():.1f}"
            )
        else:
            title = (
                f"frame={counter}  mode={display_mode}  "
                f"recv={received_fps:.1f} Hz  disp={display_fps:.1f} Hz  "
                f"min={int(display.min())}  max={int(display.max())}  "
                f"mean={display.mean():.1f}"
            )

        cv2.putText(
            img,
            title,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    else:
        img = grid_img

    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_w
            if show_header:
                y0 = header_h + r * cell_h
            else:
                y0 = r * cell_h 
            x1 = x0 + cell_w
            y1 = y0 + cell_h

            cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 40), 1)

            if show_values:
                value = int(display[r, c])
                brightness = int(norm[r, c])
                text_color = (255, 255, 255) if brightness < 130 else (0, 0, 0)

                cv2.putText(
                    img,
                    str(value),
                    (x0 + 3, y0 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    text_color,
                    1,
                    cv2.LINE_AA
                )
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/cu.usbmodem2078388A45461", help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic min/max display scaling")
    parser.add_argument("--no_display", action="store_true", help="Read frames without drawing the heatmap")
    parser.add_argument("--show_adc", action="store_true", help="Show ADC value text in each heatmap cell")
    parser.add_argument("--show_header", action="store_true", help="Show header area with stats")
    parser.add_argument(
        "--baseline_frames",
        type=int,
        default=180,
        help="Number of unloaded displayed frames used for the taxel-wise baseline",
    )
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Start in raw ADC mode without automatically capturing a baseline",
    )
    parser.add_argument(
        "--raw_vmin",
        type=float,
        default=1900.0,
        help="Lower colour limit for raw ADC display",
    )
    parser.add_argument(
        "--raw_vmax",
        type=float,
        default=4050.0,
        help="Upper colour limit for raw ADC display",
    )
    parser.add_argument(
        "--signal_vmax",
        type=float,
        default=2100.0,
        help="Upper colour limit for baseline-subtracted response",
    )
    parser.add_argument(
        "--display_threshold",
        type=float,
        default=250.0,
        help=(
            "Set positive baseline-subtracted values below this level "
            "to zero in the displayed heatmap"
        ),
    )
    parser.add_argument(
        "--snapshot_dir",
        default="tactile_snapshots_iou",
        help="Root folder in which each saved three-image snapshot set is stored",
    )
    args = parser.parse_args()

    if args.baseline_frames <= 0:
        parser.error("--baseline_frames must be positive")
    if args.raw_vmax <= args.raw_vmin:
        parser.error("--raw_vmax must be larger than --raw_vmin")
    if args.signal_vmax <= 0:
        parser.error("--signal_vmax must be positive")
    if args.display_threshold < 0:
        parser.error("--display_threshold cannot be negative")

    snapshot_root = Path(args.snapshot_dir)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    print(f"Opening {args.port}...")
    # Pyserial requires a baud setting; USB CDC throughput is usually independent of it.
    # A short timeout keeps shutdown responsive.
    ser = serial.Serial(args.port, args.baud, timeout=0.005)

    # Discard queued data so the viewer starts with current frames.
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("Reading tactile frames...")
    if args.no_display:
        print("No display mode. Press Ctrl+C to quit.")
    else:
        print("Press q to quit, s to save all three heatmap views,")
        print("b to recapture the unloaded baseline, and r to toggle raw/delta view.")
        cv2.namedWindow("Tactile Matrix 32x12", cv2.WINDOW_NORMAL)

    state = {
        "counter": None,
        "matrix": None,
        "received_frames": 0,
        "reader_dropped_frames": 0,
        "stm32_counter_skips": 0,
    }

    lock = threading.Lock()
    stop_event = threading.Event()

    reader = threading.Thread(
        target=serial_reader_thread,
        args=(ser, state, lock, stop_event),
        daemon=True,
    )
    reader.start()

    display_counter = 0
    stats_t0 = time.perf_counter()
    current_display_fps = 0.0
    current_received_fps = 0.0

    last_displayed_counter = None
    display_skipped_frames = 0

    # Store the baseline in STM32 orientation (32 columns x 12 rows).
    baseline_matrix = None
    baseline_samples = []
    collecting_baseline = not args.no_baseline
    show_delta = not args.no_baseline

    if collecting_baseline:
        print(
            f"Keep the sensor unloaded while collecting "
            f"{args.baseline_frames} baseline frames..."
        )

    try:
        while True:
            with lock:
                counter = state["counter"]
                matrix = None if state["matrix"] is None else state["matrix"].copy()

            if matrix is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            # Avoid redrawing unchanged frames when display polling is faster than acquisition.
            if counter == last_displayed_counter:
                if args.no_display:
                    time.sleep(0.001)
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                continue

            # Measure counter gaps between displayed frames, including uint32 wraparound.
            if last_displayed_counter is not None:
                expected = (last_displayed_counter + 1) & 0xFFFFFFFF
                if counter != expected:
                    display_skipped_frames += (counter - expected) & 0xFFFFFFFF

            last_displayed_counter = counter
            display_counter += 1

            # Use a per-taxel median baseline; float samples prevent unsigned subtraction wrap.
            if collecting_baseline:
                baseline_samples.append(matrix.astype(np.float32))

                if len(baseline_samples) >= args.baseline_frames:
                    baseline_matrix = np.median(
                        np.stack(baseline_samples, axis=0),
                        axis=0,
                    ).astype(np.float32)

                    baseline_samples.clear()
                    collecting_baseline = False
                    show_delta = True

                    print(
                        "Baseline captured. "
                        "Displaying baseline-subtracted response."
                    )

            now = time.perf_counter()
            elapsed = now - stats_t0

            if elapsed >= 1.0:
                current_display_fps = display_counter / elapsed

                with lock:
                    current_received_fps = state["received_frames"] / elapsed
                    reader_dropped = state["reader_dropped_frames"]
                    stm32_skips = state["stm32_counter_skips"]

                    state["received_frames"] = 0
                    state["reader_dropped_frames"] = 0
                    state["stm32_counter_skips"] = 0
    
                print(
                    f"Display rate: {current_display_fps:.2f} Hz | "
                    f"received: {current_received_fps:.2f} Hz | "
                    f"display skipped: {display_skipped_frames} | "
                    f"reader dropped old: {reader_dropped} | "
                    f"STM32 counter skips: {stm32_skips}"
                )

                display_counter = 0
                display_skipped_frames = 0
                stats_t0 = now

            if not args.no_display:
                img = matrix_to_image(
                    matrix,
                    dynamic=args.dynamic,
                    counter=counter,
                    display_fps=current_display_fps,
                    received_fps=current_received_fps,
                    show_values=args.show_adc,
                    show_header=args.show_header,
                    baseline_matrix=(
                        baseline_matrix
                        if show_delta and baseline_matrix is not None
                        else None
                    ),
                    raw_vmin=args.raw_vmin,
                    raw_vmax=args.raw_vmax,
                    signal_vmax=args.signal_vmax,
                    display_threshold=args.display_threshold,
                )

                cv2.imshow("Tactile Matrix 32x12", img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    if baseline_matrix is None:
                        print(
                            "Cannot save the processed view because no baseline "
                            "is available. Capture one first with b."
                        )
                    else:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        save_dir = snapshot_root / f"{timestamp}_frame_{counter}"
                        save_dir.mkdir(parents=True, exist_ok=False)

                        common_kwargs = dict(
                            matrix=matrix,
                            dynamic=False,
                            counter=counter,
                            display_fps=current_display_fps,
                            received_fps=current_received_fps,
                            show_values=args.show_adc,
                            show_header=args.show_header,
                        )

                        processed_img = matrix_to_image(
                            **common_kwargs,
                            baseline_matrix=baseline_matrix,
                            raw_vmin=args.raw_vmin,
                            raw_vmax=args.raw_vmax,
                            signal_vmax=args.signal_vmax,
                            display_threshold=args.display_threshold,
                        )

                        raw_focused_img = matrix_to_image(
                            **common_kwargs,
                            baseline_matrix=None,
                            raw_vmin=args.raw_vmin,
                            raw_vmax=args.raw_vmax,
                            signal_vmax=args.signal_vmax,
                            display_threshold=args.display_threshold,
                        )

                        # Retain a full 12-bit view alongside the configured display ranges.
                        raw_full_img = matrix_to_image(
                            **common_kwargs,
                            baseline_matrix=None,
                            raw_vmin=0.0,
                            raw_vmax=4095.0,
                            signal_vmax=args.signal_vmax,
                            display_threshold=args.display_threshold,
                        )

                        processed_name = (
                            f"01_processed_delta_threshold_"
                            f"{args.display_threshold:.0f}.png"
                        )
                        focused_name = (
                            f"02_raw_focused_"
                            f"{args.raw_vmin:.0f}_{args.raw_vmax:.0f}.png"
                        )
                        full_name = "03_raw_full_0_4095.png"

                        save_results = {
                            processed_name: cv2.imwrite(
                                str(save_dir / processed_name),
                                processed_img,
                            ),
                            focused_name: cv2.imwrite(
                                str(save_dir / focused_name),
                                raw_focused_img,
                            ),
                            full_name: cv2.imwrite(
                                str(save_dir / full_name),
                                raw_full_img,
                            ),
                        }

                        failed = [
                            name for name, success in save_results.items()
                            if not success
                        ]

                        if failed:
                            print(
                                "Some images could not be saved: "
                                + ", ".join(failed)
                            )
                        else:
                            print(f"Saved three heatmaps in: {save_dir}")

                elif key == ord("b"):
                    baseline_matrix = None
                    baseline_samples.clear()
                    collecting_baseline = True
                    show_delta = True

                    print(
                        f"Recapturing baseline from {args.baseline_frames} frames. "
                        "Remove all objects and keep the sensor unloaded."
                    )

                elif key == ord("r"):
                    if baseline_matrix is None:
                        show_delta = False
                        print("No baseline is available yet; displaying raw ADC values.")
                    else:
                        show_delta = not show_delta
                        print(
                            "Displaying baseline-subtracted response."
                            if show_delta
                            else "Displaying raw ADC values."
                        )

                elif key == ord("q"):
                    break
            else:
                # Yield to the reader thread instead of busy-waiting.
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stop_event.set()
        reader.join(timeout=0.5)
        ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()