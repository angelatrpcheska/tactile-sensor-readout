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
#CRC_SIZE = 2
DATA_SIZE = NUM_COLS * NUM_ROWS * 2
#NUM_VALUES = NUM_COLS * NUM_ROWS

#if NUM_VALUES % 2 != 0:
#    raise ValueError("12-bit packing requires an even number of ADC values")

#DATA_SIZE = (NUM_VALUES * 3) // 2
PACKET_SIZE = HEADER_SIZE + DATA_SIZE
#PACKET_SIZE = HEADER_SIZE + DATA_SIZE + CRC_SIZE

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

# def unpack_12bit_payload(payload):
#     """
#     Unpack STM32 format:
#       2 ADC values = 3 bytes

#     byte0 = a[7:0]
#     byte1 = a[11:8] | b[3:0] << 4
#     byte2 = b[11:4]
#     """
#     raw = np.frombuffer(payload, dtype=np.uint8)

#     if raw.size != DATA_SIZE:
#         return None

#     triples = raw.reshape(-1, 3).astype(np.uint16)

#     a = triples[:, 0] | ((triples[:, 1] & 0x0F) << 8)
#     b = ((triples[:, 1] >> 4) & 0x0F) | (triples[:, 2] << 4)

#     values = np.empty(NUM_VALUES, dtype=np.uint16)
#     values[0::2] = a & 0x0FFF
#     values[1::2] = b & 0x0FFF

#     return values


# def crc16_ccitt_false(data):
#     crc = 0xFFFF

#     for byte in data:
#         crc ^= byte << 8

#         for _ in range(8):
#             if crc & 0x8000:
#                 crc = ((crc << 1) ^ 0x1021) & 0xFFFF
#             else:
#                 crc = (crc << 1) & 0xFFFF

#     return crc


def read_frame(ser):
    """
    Blocking-ish read of one complete tactile frame.
    Kept mostly for debugging; the live viewer below uses the faster buffered reader.
    """
    # Find 0xAA 0x55
    if not find_sync(ser):
        return None, None

    # Read frame counter
    counter_bytes = read_exact(ser, 4)
    if counter_bytes is None:
        return None, None

    frame_counter = struct.unpack("<I", counter_bytes)[0]

    # Read ADC frame data
    payload = read_exact(ser, DATA_SIZE)
    if payload is None:
        return None, None

    # crc_bytes = read_exact(ser, CRC_SIZE)
    # if crc_bytes is None:
    #     return None, None

    # crc_rx = struct.unpack("<H", crc_bytes)[0]
    # crc_calc = crc16_ccitt_false(counter_bytes + payload)

    # if crc_rx != crc_calc:
    #     return None, None

    # values = unpack_12bit_payload(payload)

    values = np.frombuffer(payload, dtype="<u2")  # little-endian uint16

    # if values.size != NUM_COLS * NUM_ROWS:
    #    return None, None
    # values = unpack_12bit_payload(payload)

    if values is None:
         return None, None

    matrix = values.reshape((NUM_COLS, NUM_ROWS)).copy()

    return frame_counter, matrix


def read_latest_available_frame(ser, rxbuf):
    """
    Read all bytes currently available from serial.
    Parse all complete frames.
    Return only the newest complete frame.

    This prevents display lag by dropping old queued frames.
    """
    # Read whatever is already waiting. If nothing is waiting, read 1 byte
    # with the short serial timeout so the reader thread can stop cleanly.
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
        # Find sync bytes AA 55
        sync_idx = rxbuf.find(SYNC)

        if sync_idx < 0:
            # Keep last byte only in case it is 0xAA and next byte becomes 0x55
            if len(rxbuf) > 1:
                rxbuf[:] = rxbuf[-1:]
            break

        # Drop garbage before sync
        if sync_idx > 0:
            del rxbuf[:sync_idx]

        # Wait until a full packet is available
        if len(rxbuf) < PACKET_SIZE:
            break

        # # Extract one packet, but do not delete it yet.
        packet = bytes(rxbuf[:PACKET_SIZE])
        del rxbuf[:PACKET_SIZE]
        # counter_bytes = packet[2:6]
        # payload_start = HEADER_SIZE
        # payload_end = HEADER_SIZE + DATA_SIZE
        # payload = packet[payload_start:payload_end]
        # crc_bytes = packet[payload_end:payload_end + CRC_SIZE]

        # crc_rx = struct.unpack("<H", crc_bytes)[0]
        # crc_calc = crc16_ccitt_false(counter_bytes + payload)

        # if crc_rx != crc_calc:
        #     # False sync or corrupted packet.
        #     # Delete only the first byte and search again.
        #     del rxbuf[0:1]
        #     continue

        # # CRC is valid, now we can remove the whole packet.
        # del rxbuf[:PACKET_SIZE]

        # counter = struct.unpack("<I", counter_bytes)[0]
        # Packet layout:
        # [0:2] sync AA 55
        # [2:6] uint32 frame counter
        # [6:]  packed 12-bit ADC values: 2 values in 3 bytes
        counter = struct.unpack("<I", packet[2:6])[0]
        payload = packet[6:]

        values = np.frombuffer(payload, dtype="<u2") # uint16 ADC values
        if values.size != NUM_COLS * NUM_ROWS:
            continue

        # values = unpack_12bit_payload(payload)

        if values is None:
            continue

        latest_counter = counter
        latest_matrix = values.reshape((NUM_COLS, NUM_ROWS)).copy()
        frames_parsed += 1

    return latest_counter, latest_matrix, frames_parsed


def serial_reader_thread(ser, state, lock, stop_event):
    """
    Continuously consume serial bytes in the background.

    The reader always stores only the newest complete frame in `state`.
    This keeps the OS / pyserial receive buffer from growing while OpenCV
    is busy drawing the heatmap.
    """
    rxbuf = bytearray()
    last_received_counter = None

    while not stop_event.is_set():
        counter, matrix, parsed_count = read_latest_available_frame(ser, rxbuf)

        if matrix is None:
            continue

        stm32_skips = 0
        if last_received_counter is not None:
            #expected = (last_received_counter + 1) & 0xFFFFFFFF
            # If read_latest_available_frame() parsed N frames and returned only
            # the newest one, then a counter jump of N is normal.
            expected = (last_received_counter + parsed_count) & 0xFFFFFFFF

            if counter != expected:
                # This means the frame counter jumped between frames that the
                # Python reader actually received. If the firmware intentionally
                # drops frames when USB is busy, this can be nonzero.
                stm32_skips = (counter - expected) & 0xFFFFFFFF

        last_received_counter = counter

        with lock:
            # Intentionally overwrite the previous frame. For live viewing,
            # newest data is better than displaying every old frame.
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
    """
    matrix shape from STM32: [column][row] = 32x12
    Display: 12 rows x 32 columns
    """

    # STM32 sends [column][row] = 32x12. Transpose to display as 12x32,
    # then flip left<->right.
    raw_display = np.fliplr(matrix.T).astype(np.float32)

    if baseline_matrix is not None:
        baseline_display = np.fliplr(
            baseline_matrix.T
        ).astype(np.float32)

        # Signed taxel-wise response. Keep this unfiltered for statistics and
        # value labels; only the colour image is thresholded.
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

    # Smaller cells make OpenCV drawing much faster and reduce display lag.
    cell_w = 30
    cell_h = 25
    header_h = 40

    grid_w = cols * cell_w
    grid_h = rows * cell_h

    # Normalize for colour.
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

    # Make colored cells
    color_small = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    grid_img = cv2.resize(
        color_small,
        (grid_w, grid_h),
        interpolation=cv2.INTER_NEAREST
    )

    if show_header:
        # Add separate header area
        img = np.zeros((header_h + grid_h, grid_w, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        img[header_h:header_h + grid_h, 0:grid_w] = grid_img

        # Header text
        
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

    # Draw grid and values
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
    # For USB CDC this baud value usually does not limit throughput, but pyserial still requires it.
    # Short timeout keeps the reader thread responsive.
    ser = serial.Serial(args.port, args.baud, timeout=0.005)

    # Give the port a moment to settle, then remove old queued data so the viewer starts live.
    #time.sleep(0.2)
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

    # FPS / stats measurement
    display_counter = 0
    stats_t0 = time.perf_counter()
    current_display_fps = 0.0
    current_received_fps = 0.0

    last_displayed_counter = None
    display_skipped_frames = 0

    # Baseline is stored in the original STM32 orientation: 32 columns x 12 rows.
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

            # No complete frame has arrived yet.
            if matrix is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            # Do not redraw the same frame over and over if display loop is faster than serial.
            if counter == last_displayed_counter:
                if args.no_display:
                    time.sleep(0.001)
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                continue

            # Count frames skipped by the display because a newer frame replaced old ones.
            if last_displayed_counter is not None:
                expected = (last_displayed_counter + 1) & 0xFFFFFFFF
                if counter != expected:
                    display_skipped_frames += (counter - expected) & 0xFFFFFFFF

            last_displayed_counter = counter
            display_counter += 1

            # Capture a robust taxel-wise baseline. Cast to float before storing
            # so later subtraction cannot wrap around as unsigned integers.
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

            # Print stats once per second.
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

                        # 1) Baseline-subtracted and thresholded contact response.
                        processed_img = matrix_to_image(
                            **common_kwargs,
                            baseline_matrix=baseline_matrix,
                            raw_vmin=args.raw_vmin,
                            raw_vmax=args.raw_vmax,
                            signal_vmax=args.signal_vmax,
                            display_threshold=args.display_threshold,
                        )

                        # 2) Raw ADC values with the focused 1900..4050-style range.
                        raw_focused_img = matrix_to_image(
                            **common_kwargs,
                            baseline_matrix=None,
                            raw_vmin=args.raw_vmin,
                            raw_vmax=args.raw_vmax,
                            signal_vmax=args.signal_vmax,
                            display_threshold=args.display_threshold,
                        )

                        # 3) Raw ADC values over the complete 12-bit range.
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
                # In no-display mode, let the reader thread do the work and avoid a CPU spin loop.
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