import argparse
import struct
import serial
import numpy as np
import cv2
import time
import threading

NUM_COLS = 32
NUM_ROWS = 12

SYNC = b"\xAA\x55"
HEADER_SIZE = 6
CRC_SIZE = 2
#DATA_SIZE = NUM_COLS * NUM_ROWS * 2
NUM_VALUES = NUM_COLS * NUM_ROWS

if NUM_VALUES % 2 != 0:
    raise ValueError("12-bit packing requires an even number of ADC values")

DATA_SIZE = (NUM_VALUES * 3) // 2
# PACKET_SIZE = HEADER_SIZE + DATA_SIZE
PACKET_SIZE = HEADER_SIZE + DATA_SIZE + CRC_SIZE

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

def unpack_12bit_payload(payload):
    """
    Unpack STM32 format:
      2 ADC values = 3 bytes

    byte0 = a[7:0]
    byte1 = a[11:8] | b[3:0] << 4
    byte2 = b[11:4]
    """
    raw = np.frombuffer(payload, dtype=np.uint8)

    if raw.size != DATA_SIZE:
        return None

    triples = raw.reshape(-1, 3).astype(np.uint16)

    a = triples[:, 0] | ((triples[:, 1] & 0x0F) << 8)
    b = ((triples[:, 1] >> 4) & 0x0F) | (triples[:, 2] << 4)

    values = np.empty(NUM_VALUES, dtype=np.uint16)
    values[0::2] = a & 0x0FFF
    values[1::2] = b & 0x0FFF

    return values

def crc16_ccitt_false(data):
    crc = 0xFFFF

    for byte in data:
        crc ^= byte << 8

        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc


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

    crc_bytes = read_exact(ser, CRC_SIZE)
    if crc_bytes is None:
        return None, None

    crc_rx = struct.unpack("<H", crc_bytes)[0]
    crc_calc = crc16_ccitt_false(counter_bytes + payload)

    if crc_rx != crc_calc:
        return None, None

    values = unpack_12bit_payload(payload)

    # values = np.frombuffer(payload, dtype="<u2")  # little-endian uint16

    # if values.size != NUM_COLS * NUM_ROWS:
    #    return None, None
    values = unpack_12bit_payload(payload)

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

        # Extract one packet, but do not delete it yet.
        packet = bytes(rxbuf[:PACKET_SIZE])
        # del rxbuf[:PACKET_SIZE]
        counter_bytes = packet[2:6]
        payload_start = HEADER_SIZE
        payload_end = HEADER_SIZE + DATA_SIZE
        payload = packet[payload_start:payload_end]
        crc_bytes = packet[payload_end:payload_end + CRC_SIZE]

        crc_rx = struct.unpack("<H", crc_bytes)[0]
        crc_calc = crc16_ccitt_false(counter_bytes + payload)

        if crc_rx != crc_calc:
            # False sync or corrupted packet.
            # Delete only the first byte and search again.
            del rxbuf[0:1]
            continue

        # CRC is valid, now we can remove the whole packet.
        del rxbuf[:PACKET_SIZE]

        counter = struct.unpack("<I", counter_bytes)[0]
        # Packet layout:
        # [0:2] sync AA 55
        # [2:6] uint32 frame counter
        # [6:]  packed 12-bit ADC values: 2 values in 3 bytes
        #counter = struct.unpack("<I", packet[2:6])[0]
        #payload = packet[6:]

        # values = np.frombuffer(payload, dtype="<u2") # uint16 ADC values
        # if values.size != NUM_COLS * NUM_ROWS:
        #     continue

        values = unpack_12bit_payload(payload)

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
):
    """
    matrix shape from STM32: [column][row] = 32x12
    Display: 12 rows x 32 columns
    """

    # STM32 sends [column][row] = 32x12. Transpose to display as 12x32,
    # then flip left<->right.
    display = np.fliplr(matrix.T).astype(np.float32)

    rows, cols = display.shape

    # Smaller cells make OpenCV drawing much faster and reduce display lag.
    cell_w = 30
    cell_h = 25
    header_h = 40

    grid_w = cols * cell_w
    grid_h = rows * cell_h

    # Normalize for color
    if dynamic:
        mn = display.min()
        mx = display.max()
        if mx > mn:
            norm = (display - mn) * 255.0 / (mx - mn)
        else:
            norm = np.zeros_like(display)
    else:
        norm = display * 255.0 / 4095.0

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
                f"recv={received_fps:.1f} Hz  disp={display_fps:.1f} Hz  "
                f"min={int(display.min())}  max={int(display.max())}  mean={display.mean():.1f}"
            )
        else:
            title = (
                f"frame={counter}  recv={received_fps:.1f} Hz  disp={display_fps:.1f} Hz  "
                f"min={int(display.min())}  max={int(display.max())}  mean={display.mean():.1f}"
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
    parser.add_argument("--port", default="/dev/cu.usbmodem2077388A45461", help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic min/max display scaling")
    parser.add_argument("--no_display", action="store_true", help="Read frames without drawing the heatmap")
    parser.add_argument("--show_adc", action="store_true", help="Show ADC value text in each heatmap cell")
    parser.add_argument("--show_header", action="store_true", help="Show header area with stats ")
    args = parser.parse_args()

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
        print("Press q in the image window to quit.")
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
                    show_header=args.show_header
                )

                cv2.imshow("Tactile Matrix 32x12", img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    cv2.imwrite(f"tactile_snapshot_{counter}.png", img)
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
