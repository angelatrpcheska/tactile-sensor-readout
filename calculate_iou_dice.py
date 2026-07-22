#!/usr/bin/env python3
"""
Calculate IoU and Dice for processed tactile heatmap snapshots against
the diagonal triple-stripe mold reference mask.

Expected folder structure:

project_directory/
├── calculate_iou_dice.py
└── tactile_snapshots_iou/
    ├── 20260720_153402_frame_222569/
    │   └── 01_processed_delta_threshold_500.png
    ├── 20260720_153854_frame_251791/
    │   └── 01_processed_delta_threshold_500.png
    └── ...

Run:
    python3 calculate_iou_dice.py

Or specify another folder:
    python3 calculate_iou_dice.py --root /path/to/tactile_snapshots_iou

Outputs are written inside tactile_snapshots_iou:
    iou_dice_results.csv
    iou_dice_summary.json
    reference_mask.png
    iou_dice_overlays/
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Sensor and saved-image configuration
# ---------------------------------------------------------------------------

ROWS = 12
COLS = 32

# The saved processed images produced by the tactile viewer are 960 x 340 px:
# 40 px title/header + 300 px heatmap.
DEFAULT_HEATMAP_TOP_PX = 40

# A taxel is considered active when its sampled colour differs sufficiently
# from the dominant zero-response colour. The processed image has already had
# responses below 500 ADC counts set to zero.
DEFAULT_COLOUR_DISTANCE_THRESHOLD = 20.0

# Use only the centre of each plotted cell, avoiding grid lines.
CELL_SAMPLE_MARGIN = 0.28


# ---------------------------------------------------------------------------
# Physical reference-mask configuration
# ---------------------------------------------------------------------------

TAXEL_PITCH_MM = 2.0
ACTIVE_WIDTH_MM = 64.0
ACTIVE_HEIGHT_MM = 24.0

# The printed mold footprint is 65 x 26 mm and is assumed to be centred over
# the 64 x 24 mm sensor area.
MOLD_WIDTH_MM = 65.0
MOLD_HEIGHT_MM = 26.0
ACTIVE_ORIGIN_X_MM = (MOLD_WIDTH_MM - ACTIVE_WIDTH_MM) / 2.0   # 0.5 mm
ACTIVE_ORIGIN_Y_MM = (MOLD_HEIGHT_MM - ACTIVE_HEIGHT_MM) / 2.0 # 1.0 mm

# Actual measured stripe width. Change this only if you re-measure the mold.
DEFAULT_STRIPE_WIDTH_MM = 4

# Fraction of a 2 x 2 mm taxel area that must be covered by the ideal stripe
# geometry for that taxel to belong to the binary reference mask.
DEFAULT_MIN_TAXEL_OVERLAP = 0.25

# Centre-line endpoints of the three diagonal stripes, in millimetres relative
# to the top-left corner of the 65 x 26 mm mold footprint.
#
# These positions come from the diagonal three-stripe CAD geometry. Only the
# stripe width is changed from the nominal design to the measured 4 mm.
STRIPE_CENTRE_LINES_MM = (
    ((8.3578643763,  6.63603896935), (21.0857864377, 19.36396103070)),
    ((26.1360389694, 6.63603896935), (38.8639610307, 19.36396103070)),
    ((43.9142135624, 6.63603896935), (56.6421356238, 19.36396103070)),
)


@dataclass(frozen=True)
class Metrics:
    intersection: int
    union: int
    measured_positive: int
    reference_positive: int
    iou: float
    dice: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate IoU and Dice for tactile snapshot PNG files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "tactile_snapshots_iou",
        help="Folder containing the timestamped snapshot subfolders.",
    )
    parser.add_argument(
        "--filename",
        default="01_processed_delta_threshold_500.png",
        help="Processed image filename to find inside each sample folder.",
    )
    parser.add_argument(
        "--reference-mask",
        type=Path,
        default=None,
        help=(
            "Optional external 12x32 binary reference mask (.npy or a clean "
            "mask PNG). When omitted, the script generates its built-in "
            "geometric triple-stripe mask."
        ),
    )
    parser.add_argument(
        "--heatmap-top",
        type=int,
        default=DEFAULT_HEATMAP_TOP_PX,
        help="Number of pixels above the 12x32 heatmap in each saved PNG.",
    )
    parser.add_argument(
        "--colour-distance",
        type=float,
        default=DEFAULT_COLOUR_DISTANCE_THRESHOLD,
        help="RGB distance from the zero-response colour required for activation.",
    )
    parser.add_argument(
        "--stripe-width",
        type=float,
        default=DEFAULT_STRIPE_WIDTH_MM,
        help="Measured physical stripe width in millimetres.",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=DEFAULT_MIN_TAXEL_OVERLAP,
        help="Minimum ideal stripe/taxel area overlap used for the reference mask.",
    )
    parser.add_argument(
        "--row-shift",
        type=int,
        default=0,
        help="Apply one fixed row shift to the reference mask; no wraparound.",
    )
    parser.add_argument(
        "--col-shift",
        type=int,
        default=0,
        help="Apply one fixed column shift to the reference mask; no wraparound.",
    )
    parser.add_argument(
        "--flip-ud",
        action="store_true",
        help="Flip the reference mask vertically to match display orientation.",
    )
    parser.add_argument(
        "--flip-lr",
        action="store_true",
        help="Flip the reference mask horizontally to match display orientation.",
    )
    parser.add_argument(
        "--supersampling",
        type=int,
        default=40,
        help="Samples per taxel dimension when rasterising the physical mold.",
    )
    return parser.parse_args()


def find_images(root: Path, filename: str) -> list[Path]:
    """Find one processed image in every timestamped sample folder."""
    if not root.exists():
        raise FileNotFoundError(f"Snapshot folder does not exist: {root}")

    paths = sorted(root.rglob(filename), key=lambda p: p.parent.name)
    if not paths:
        raise FileNotFoundError(
            f"No files named {filename!r} were found below {root}"
        )
    return paths


def _cell_centre_colours(
    image_rgb: np.ndarray,
    heatmap_top: int,
) -> np.ndarray:
    """
    Return one representative RGB colour per taxel by taking the median over
    the centre region of each plotted heatmap cell.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image, received shape {image_rgb.shape}")

    height, width, _ = image_rgb.shape
    if heatmap_top < 0 or heatmap_top >= height:
        raise ValueError(
            f"Invalid heatmap-top={heatmap_top} for image height {height}"
        )

    heatmap = image_rgb[heatmap_top:, :, :]
    heat_h, heat_w, _ = heatmap.shape

    cell_h = heat_h / ROWS
    cell_w = heat_w / COLS

    colours = np.zeros((ROWS, COLS, 3), dtype=np.float32)

    for row in range(ROWS):
        for col in range(COLS):
            y0 = int(round((row + CELL_SAMPLE_MARGIN) * cell_h))
            y1 = int(round((row + 1.0 - CELL_SAMPLE_MARGIN) * cell_h))
            x0 = int(round((col + CELL_SAMPLE_MARGIN) * cell_w))
            x1 = int(round((col + 1.0 - CELL_SAMPLE_MARGIN) * cell_w))

            y0 = max(0, min(y0, heat_h - 1))
            y1 = max(y0 + 1, min(y1, heat_h))
            x0 = max(0, min(x0, heat_w - 1))
            x1 = max(x0 + 1, min(x1, heat_w))

            patch = heatmap[y0:y1, x0:x1]
            colours[row, col] = np.median(
                patch.reshape(-1, 3), axis=0
            )

    return colours


def extract_measured_mask(
    image_path: Path,
    heatmap_top: int,
    colour_distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover the 12x32 binary mask from a processed heatmap PNG.

    Returns:
        measured_mask: bool array, shape (12, 32)
        cell_colours: representative RGB value for every taxel
        background_rgb: dominant zero-response RGB colour
    """
    image = np.asarray(Image.open(image_path).convert("RGB"))
    colours = _cell_centre_colours(image, heatmap_top)

    # Quantise slightly, then use the most frequent taxel colour as the
    # zero-response background. Most taxels should be inactive.
    quantised = (np.rint(colours / 4.0) * 4.0).astype(np.int16)
    flat = quantised.reshape(-1, 3)
    unique, counts = np.unique(flat, axis=0, return_counts=True)
    background_rgb = unique[int(np.argmax(counts))].astype(np.float32)

    distances = np.linalg.norm(colours - background_rgb[None, None, :], axis=2)
    measured_mask = distances > colour_distance_threshold

    return measured_mask, colours, background_rgb



def load_reference_mask_file(mask_path: Path) -> np.ndarray:
    """
    Load an external binary 12x32 reference mask.

    Recommended format:
        np.save("dxf_reference_mask_4p2mm.npy", mask.astype(np.uint8))

    Supported inputs:
      - .npy containing an array with shape (12, 32)
      - a clean black/white PNG with shape 12x32
      - a clean enlarged PNG whose dimensions are exact multiples of 12x32

    Do not pass a Matplotlib screenshot containing a title, axes, margins, or
    labels. Save the underlying binary matrix instead.
    """
    path = mask_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference mask does not exist: {path}")

    if path.suffix.lower() == ".npy":
        array = np.load(path)
        array = np.squeeze(array)
        if array.shape != (ROWS, COLS):
            raise ValueError(
                f"Reference .npy must have shape {(ROWS, COLS)}, "
                f"but {path.name} has shape {array.shape}"
            )
        return array.astype(np.float32) > 0.5

    image = np.asarray(Image.open(path).convert("L"))
    height, width = image.shape

    if (height, width) == (ROWS, COLS):
        cell_values = image.astype(np.float32)
    elif (
        height % ROWS == 0
        and width % COLS == 0
        and width * ROWS == height * COLS
    ):
        # Average each clean enlarged taxel block. This tolerates thin grid
        # lines but assumes the PNG contains only the mask, with no axes/title.
        block_h = height // ROWS
        block_w = width // COLS
        cell_values = image.reshape(
            ROWS, block_h, COLS, block_w
        ).mean(axis=(1, 3))
    else:
        raise ValueError(
            "The reference PNG must be a clean 12x32 mask or an exact "
            "integer enlargement of it. Do not use the plotted figure with "
            "title and axes. Save the underlying 12x32 binary array as .npy."
        )

    # Works for either 0/1 or 0/255 images.
    threshold = 0.5 if float(cell_values.max()) <= 1.0 else 127.5
    return cell_values > threshold


def _inside_flat_capped_stripe(
    x: np.ndarray,
    y: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    width_mm: float,
) -> np.ndarray:
    """
    Test whether points are inside a finite rectangular stripe with flat ends.
    """
    x0, y0 = start_xy
    x1, y1 = end_xy

    vx = x1 - x0
    vy = y1 - y0
    length_sq = vx * vx + vy * vy
    length = float(np.sqrt(length_sq))

    px = x - x0
    py = y - y0

    t = (px * vx + py * vy) / length_sq
    perpendicular_distance = np.abs(vx * py - vy * px) / length

    return (
        (t >= 0.0)
        & (t <= 1.0)
        & (perpendicular_distance <= width_mm / 2.0)
    )


def build_reference_mask(
    stripe_width_mm: float,
    min_taxel_overlap: float,
    supersampling: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rasterise the physical triple-stripe geometry onto the 12x32 taxel grid.

    Returns:
        reference_mask: bool array, shape (12, 32)
        overlap_fractions: ideal stripe coverage fraction for every taxel
    """
    if stripe_width_mm <= 0:
        raise ValueError("stripe_width_mm must be positive")
    if not 0.0 <= min_taxel_overlap <= 1.0:
        raise ValueError("min_taxel_overlap must be in [0, 1]")
    if supersampling < 2:
        raise ValueError("supersampling must be at least 2")

    # Sample at sub-taxel centres.
    sub = (np.arange(supersampling, dtype=np.float64) + 0.5) / supersampling

    overlap_fractions = np.zeros((ROWS, COLS), dtype=np.float64)

    for row in range(ROWS):
        for col in range(COLS):
            x = (
                ACTIVE_ORIGIN_X_MM
                + (col + sub[None, :]) * TAXEL_PITCH_MM
            )
            y = (
                ACTIVE_ORIGIN_Y_MM
                + (row + sub[:, None]) * TAXEL_PITCH_MM
            )

            # Broadcast to a supersampling x supersampling point grid.
            x_grid = np.broadcast_to(x, (supersampling, supersampling))
            y_grid = np.broadcast_to(y, (supersampling, supersampling))

            inside_any = np.zeros(
                (supersampling, supersampling), dtype=bool
            )
            for start_xy, end_xy in STRIPE_CENTRE_LINES_MM:
                inside_any |= _inside_flat_capped_stripe(
                    x_grid,
                    y_grid,
                    start_xy,
                    end_xy,
                    stripe_width_mm,
                )

            overlap_fractions[row, col] = float(np.mean(inside_any))

    reference_mask = overlap_fractions >= min_taxel_overlap
    return reference_mask, overlap_fractions


def shift_mask(mask: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    """Shift without wraparound; newly exposed cells are False."""
    shifted = np.zeros_like(mask, dtype=bool)

    source_r0 = max(0, -row_shift)
    source_r1 = min(ROWS, ROWS - row_shift)
    source_c0 = max(0, -col_shift)
    source_c1 = min(COLS, COLS - col_shift)

    target_r0 = max(0, row_shift)
    target_r1 = min(ROWS, ROWS + row_shift)
    target_c0 = max(0, col_shift)
    target_c1 = min(COLS, COLS + col_shift)

    if source_r0 < source_r1 and source_c0 < source_c1:
        shifted[target_r0:target_r1, target_c0:target_c1] = (
            mask[source_r0:source_r1, source_c0:source_c1]
        )
    return shifted


def orient_reference_mask(
    mask: np.ndarray,
    flip_ud: bool,
    flip_lr: bool,
    row_shift: int,
    col_shift: int,
) -> np.ndarray:
    result = mask.copy()
    if flip_ud:
        result = np.flipud(result)
    if flip_lr:
        result = np.fliplr(result)
    result = shift_mask(result, row_shift, col_shift)
    return result


def calculate_metrics(
    measured_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> Metrics:
    measured = measured_mask.astype(bool)
    reference = reference_mask.astype(bool)

    intersection = int(np.logical_and(measured, reference).sum())
    union = int(np.logical_or(measured, reference).sum())
    measured_positive = int(measured.sum())
    reference_positive = int(reference.sum())

    iou = intersection / union if union else 1.0
    denominator = measured_positive + reference_positive
    dice = (2.0 * intersection / denominator) if denominator else 1.0

    return Metrics(
        intersection=intersection,
        union=union,
        measured_positive=measured_positive,
        reference_positive=reference_positive,
        iou=float(iou),
        dice=float(dice),
    )


def mask_to_image(
    mask: np.ndarray,
    scale: int = 24,
    active_colour: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    rgb = np.zeros((ROWS, COLS, 3), dtype=np.uint8)
    rgb[mask] = active_colour
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((COLS * scale, ROWS * scale), Image.Resampling.NEAREST)


def save_reference_image(reference_mask: np.ndarray, output_path: Path) -> None:
    image = mask_to_image(reference_mask)
    draw = ImageDraw.Draw(image)
    scale = image.width // COLS

    for col in range(COLS + 1):
        x = col * scale
        draw.line((x, 0, x, image.height), fill=(70, 70, 70), width=1)
    for row in range(ROWS + 1):
        y = row * scale
        draw.line((0, y, image.width, y), fill=(70, 70, 70), width=1)

    image.save(output_path)


def save_overlay(
    measured_mask: np.ndarray,
    reference_mask: np.ndarray,
    output_path: Path,
    scale: int = 24,
) -> None:
    """
    Overlay colours:
        green  = overlap / true positive
        red    = measured only / false positive
        blue   = reference only / false negative
        black  = neither
    """
    measured = measured_mask.astype(bool)
    reference = reference_mask.astype(bool)

    rgb = np.zeros((ROWS, COLS, 3), dtype=np.uint8)
    rgb[np.logical_and(measured, reference)] = (0, 210, 0)
    rgb[np.logical_and(measured, ~reference)] = (235, 45, 45)
    rgb[np.logical_and(~measured, reference)] = (50, 110, 255)

    image = Image.fromarray(rgb, mode="RGB")
    image = image.resize((COLS * scale, ROWS * scale), Image.Resampling.NEAREST)

    draw = ImageDraw.Draw(image)
    for col in range(COLS + 1):
        x = col * scale
        draw.line((x, 0, x, image.height), fill=(80, 80, 80), width=1)
    for row in range(ROWS + 1):
        y = row * scale
        draw.line((0, y, image.width, y), fill=(80, 80, 80), width=1)

    image.save(output_path)


def _statistics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sequence")

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "median": float(np.median(array)),
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    image_paths = find_images(root, args.filename)

    if args.reference_mask is not None:
        reference_source = str(args.reference_mask.expanduser().resolve())
        reference_mask = load_reference_mask_file(args.reference_mask)
    else:
        reference_source = "built-in geometric triple-stripe mask"
        reference_mask, _ = build_reference_mask(
            stripe_width_mm=args.stripe_width,
            min_taxel_overlap=args.min_overlap,
            supersampling=args.supersampling,
        )

    reference_mask = orient_reference_mask(
        reference_mask,
        flip_ud=args.flip_ud,
        flip_lr=args.flip_lr,
        row_shift=args.row_shift,
        col_shift=args.col_shift,
    )

    output_csv = root / "iou_dice_results.csv"
    output_json = root / "iou_dice_summary.json"
    reference_png = root / "reference_mask.png"
    overlay_dir = root / "iou_dice_overlays"
    overlay_dir.mkdir(exist_ok=True)

    save_reference_image(reference_mask, reference_png)

    rows_out: list[dict[str, object]] = []

    print(f"Found {len(image_paths)} processed snapshots")
    print(
        f"Reference mask: {int(reference_mask.sum())} active taxels"
    )
    print(f"Reference source: {reference_source}")
    print()

    for image_path in image_paths:
        try:
            measured_mask, _, background_rgb = extract_measured_mask(
                image_path=image_path,
                heatmap_top=args.heatmap_top,
                colour_distance_threshold=args.colour_distance,
            )
            metrics = calculate_metrics(measured_mask, reference_mask)

            sample_name = image_path.parent.name
            overlay_path = overlay_dir / f"{sample_name}_overlay.png"
            save_overlay(measured_mask, reference_mask, overlay_path)

            result = {
                "sample": sample_name,
                "image": str(image_path),
                "measured_active_taxels": metrics.measured_positive,
                "reference_active_taxels": metrics.reference_positive,
                "intersection": metrics.intersection,
                "union": metrics.union,
                "iou": metrics.iou,
                "dice": metrics.dice,
                "background_r": int(round(background_rgb[0])),
                "background_g": int(round(background_rgb[1])),
                "background_b": int(round(background_rgb[2])),
            }
            rows_out.append(result)

            print(
                f"{sample_name}: "
                f"IoU={metrics.iou:.4f}, Dice={metrics.dice:.4f}, "
                f"measured={metrics.measured_positive}, "
                f"intersection={metrics.intersection}"
            )

        except Exception as exc:
            print(f"[ERROR] {image_path}: {exc}")

    if not rows_out:
        raise RuntimeError("No snapshot could be processed successfully")

    fieldnames = list(rows_out[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    iou_stats = _statistics(float(row["iou"]) for row in rows_out)
    dice_stats = _statistics(float(row["dice"]) for row in rows_out)

    summary = {
        "number_of_samples": len(rows_out),
        "source_root": str(root),
        "processed_filename": args.filename,
        "reference_mask": {
            "source": reference_source,
            "rows": ROWS,
            "columns": COLS,
            "active_taxels": int(reference_mask.sum()),
            "taxel_pitch_mm": TAXEL_PITCH_MM,
            "active_area_mm": [ACTIVE_WIDTH_MM, ACTIVE_HEIGHT_MM],
            "mold_footprint_mm": [MOLD_WIDTH_MM, MOLD_HEIGHT_MM],
            "measured_stripe_width_mm": args.stripe_width,
            "minimum_taxel_overlap": args.min_overlap,
            "flip_up_down": args.flip_ud,
            "flip_left_right": args.flip_lr,
            "row_shift": args.row_shift,
            "column_shift": args.col_shift,
        },
        "png_mask_extraction": {
            "heatmap_top_px": args.heatmap_top,
            "colour_distance_threshold": args.colour_distance,
        },
        "iou": iou_stats,
        "dice": dice_stats,
    }

    output_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSummary")
    print("-------")
    print(
        f"IoU:  mean={iou_stats['mean']:.4f}, "
        f"std={iou_stats['std']:.4f}, "
        f"min={iou_stats['min']:.4f}, "
        f"max={iou_stats['max']:.4f}"
    )
    print(
        f"Dice: mean={dice_stats['mean']:.4f}, "
        f"std={dice_stats['std']:.4f}, "
        f"min={dice_stats['min']:.4f}, "
        f"max={dice_stats['max']:.4f}"
    )
    print()
    print(f"Saved per-sample results: {output_csv}")
    print(f"Saved summary:            {output_json}")
    print(f"Saved reference mask:     {reference_png}")
    print(f"Saved overlays:           {overlay_dir}")
    print()
    print("Overlay colours: green=overlap, red=measured only, blue=reference only.")


if __name__ == "__main__":
    main()
