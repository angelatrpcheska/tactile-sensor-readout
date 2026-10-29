#!/usr/bin/env python3
"""Regenerate Slovenian thesis figures from an EXISTING noise_results.npz.

No serial port, sensor connection, or new acquisition is required.

Examples:
    python plot_tactile_results.py "/path/to/noise_results.npz" --duration-sec 180
    python plot_tactile_results.py "/path/to/noise_results.npz" --duration-sec 180 --all --timeseries
    python plot_tactile_results.py "/path/to/results_folder" --all --no-title

Each figure is saved as both PNG and PDF. Use --font-size to adjust text size.

Requirements: pip install numpy matplotlib
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Works without a graphical desktop.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def find_npz(path: Path) -> Path:
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError("Please supply a .npz file (usually noise_results.npz).")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"File or folder not found: {path}")
    exact = path / "noise_results.npz"
    if exact.exists():
        return exact
    candidates = sorted(path.glob("noise_results*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No noise_results*.npz found inside {path}")
    raise ValueError("Several NPZ results found. Supply an exact file path:\n" +
                     "\n".join(f"  {file}" for file in candidates))


def read_matrix(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing '{key}' in NPZ. Available keys: {', '.join(data.files)}")
    array = np.asarray(data[key], dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D matrix for {key}, got {array.shape}")
    return array


def top_rounded(value: float, step: float, minimum: float) -> float:
    return max(minimum, math.ceil(float(value) / step) * step)


def save_figure(fig, filename: Path) -> None:
    for suffix in (".png", ".pdf"):
        output = filename.with_suffix(suffix)
        fig.savefig(output, dpi=250, bbox_inches="tight")
        print(f"Saved: {output}")
    plt.close(fig)


def make_heatmap(matrix, filename: Path, title: str, color_label: str,
                 vmin: float, vmax: float, cmap: str = "viridis",
                 show_title: bool = True) -> None:
    rows, cols = matrix.shape
    fig, ax = plt.subplots(figsize=(10.6, 4.6), layout="constrained")
    img = ax.imshow(matrix, cmap=cmap, origin="upper", aspect="auto",
                    interpolation="nearest", vmin=vmin, vmax=vmax,
                    extent=(0.5, cols + 0.5, rows + 0.5, 0.5))
    if show_title:
        ax.set_title(title)
    ax.set_xlabel("Stolpec")
    ax.set_ylabel("Vrstica")
    ax.set_xticks(sorted(set([1, cols] + list(range(4, cols + 1, 4)))))
    ax.set_yticks(sorted(set([1, rows] + list(range(2, rows + 1, 2)))))
    cb = fig.colorbar(img, ax=ax, pad=0.02, shrink=0.90)
    cb.set_label(color_label)
    save_figure(fig, filename)


def make_saturation_mask(mean: np.ndarray, filename: Path,
                         adc_max: int, margin: int, show_title: bool = True) -> None:
    mask = np.zeros_like(mean, dtype=np.int8)
    mask[mean < margin] = -1
    mask[mean > adc_max - margin] = 1
    rows, cols = mask.shape
    cmap = ListedColormap(["#4477aa", "#dddddd", "#bb5566"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10.6, 4.6), layout="constrained")
    img = ax.imshow(mask, cmap=cmap, norm=norm, origin="upper", aspect="auto",
                    interpolation="nearest", extent=(0.5, cols + 0.5, rows + 0.5, 0.5))
    if show_title:
        ax.set_title("Taksli blizu mej pretvornika ADC (izhodiščne vrednosti)")
    ax.set_xlabel("Stolpec")
    ax.set_ylabel("Vrstica")
    ax.set_xticks(sorted(set([1, cols] + list(range(4, cols + 1, 4)))))
    ax.set_yticks(sorted(set([1, rows] + list(range(2, rows + 1, 2)))))
    cbar = fig.colorbar(img, ax=ax, ticks=[-1, 0, 1], shrink=0.90, pad=0.02)
    cbar.ax.set_yticklabels(["−1: nizka", "0: brez", "+1: visoka"])
    cbar.set_label("Bližina mejnega območja ADC")
    save_figure(fig, filename)


def make_timeseries(data: np.lib.npyio.NpzFile, filename: Path,
                    duration_sec: Optional[float], show_title: bool = True) -> None:
    if "baseline_frames" not in data:
        print("Time-series skipped: saved NPZ has no 'baseline_frames'.")
        return
    frames = data["baseline_frames"]
    if frames.ndim != 3:
        raise ValueError(f"Expected baseline_frames of shape (N, rows, cols), got {frames.shape}")
    if frames.shape[1] < 1 or frames.shape[2] < 20:
        print("Time-series skipped: expected at least 1 row and 20 columns.")
        return
    # Zero-based coordinates. Changes near (0,17) and (0,18) were seen in
    # the 180-second trial; (0,19) is included as a nearby stable comparison.
    chosen = [(0, 17), (0, 18), (0, 19)]
    sample_step = max(1, math.ceil(len(frames) / 3500))
    indices = np.arange(0, len(frames), sample_step)
    if duration_sec is None:
        x = indices
        xlabel = "Zaporedna številka okvirja"
    else:
        x = indices * (duration_sec / len(frames))
        xlabel = "Čas [s]"
    fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
    for row, col in chosen:
        # Baseline of each taxel is subtracted for visual comparability.
        signal = np.asarray(frames[indices, row, col], dtype=float)
        reference = float(np.median(frames[:min(len(frames), 3000), row, col]))
        ax.plot(x, signal - reference, linewidth=1.0,
                label=f"Vrstica {row + 1}, stolpec {col + 1}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sprememba glede na začetno raven\n[vrednosti ADC]")
    if show_title:
        ax.set_title("Časovni potek signala izbranih sosednjih takslov")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    save_figure(fig, filename)


def main():
    parser = argparse.ArgumentParser(description="Plot saved tactile measurements without recording again.")
    parser.add_argument("results", type=Path, help="Path to noise_results.npz or its containing folder")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: thesis_figures next to NPZ)")
    parser.add_argument("--duration-sec", type=float, default=None,
                        help="Recording duration, e.g. 180, for figure titles and time axis")
    parser.add_argument("--std-vmax", type=float, default=35.0,
                        help="Upper colorbar bound for standard deviation (default: 35 ADC)")
    parser.add_argument("--all", action="store_true",
                        help="Regenerate all six types of saved PNG and PDF figures")
    parser.add_argument("--timeseries", action="store_true",
                        help="Also plot neighboring taxels from saved raw frames")
    parser.add_argument("--no-title", action="store_true",
                        help="Omit titles from all figures")
    parser.add_argument("--font-size", type=float, default=14.0,
                        help="Base text size in points (default: 14)")
    parser.add_argument("--adc-max", type=int, default=4095)
    parser.add_argument("--saturation-margin", type=int, default=20)
    args = parser.parse_args()
    if not math.isfinite(args.font_size) or args.font_size <= 0:
        parser.error("--font-size must be a positive finite number")
    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.titlesize": args.font_size + 2,
        "axes.labelsize": args.font_size,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size,
        "legend.fontsize": args.font_size,
    })

    source = find_npz(args.results.expanduser())
    destination = args.out_dir.expanduser() if args.out_dir else source.parent / "thesis_figures"
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Reading existing results: {source}")
    print("No new sensor measurements will be performed.")
    with np.load(source, allow_pickle=False) as npz:
        std = read_matrix(npz, "noise_std")
        max_std = float(np.nanmax(std))
        std_vmax = max(args.std_vmax, top_rounded(max_std, 5.0, 5.0))
        duration_text = f" ({args.duration_sec:g} s)" if args.duration_sec else ""
        make_heatmap(std, destination / "standardni_odklon.png",
                     "Časovni standardni odklon brez obremenitve" + duration_text,
                     "Standardni odklon [vrednosti ADC]", 0.0, std_vmax, show_title=not args.no_title)
        print(f"STD median: {np.nanmedian(std):.3f} ADC; max: {max_std:.3f} ADC; display scale: 0–{std_vmax:g}")

        if args.all:
            mean = read_matrix(npz, "baseline_mean")
            p2p = read_matrix(npz, "noise_p2p")
            drift = read_matrix(npz, "baseline_abs_drift")
            low = math.floor(float(np.nanmin(mean)) / 20) * 20
            high = math.ceil(float(np.nanmax(mean)) / 20) * 20
            make_heatmap(mean, destination / "povprecna_izhodiscna_vrednost_adc.png",
                         "Povprečna izhodiščna vrednost (celotni razpon ADC)",
                         "Povprečna vrednost [vrednosti ADC]", 0, args.adc_max, show_title=not args.no_title)
            make_heatmap(mean, destination / "povprecna_izhodiscna_vrednost_povecava.png",
                         "Povprečna izhodiščna vrednost (povečava)",
                         "Povprečna vrednost [vrednosti ADC]", low, max(high, low + 20),
                         show_title=not args.no_title)
            make_heatmap(p2p, destination / "razpon_najmanj_najvec.png",
                         "Razlika med največjo in najmanjšo vrednostjo",
                         "Razpon signala [vrednosti ADC]", 0,
                         top_rounded(float(np.nanmax(p2p)), 10, 10),
                         show_title=not args.no_title)
            make_heatmap(drift, destination / "absolutna_sprememba_izhodiscnega_signala.png",
                         "Absolutna sprememba povprečja med polovicama meritve",
                         "|Razlika povprečij| [vrednosti ADC]", 0,
                         top_rounded(float(np.nanmax(drift)), 1, 1),
                         show_title=not args.no_title)
            make_saturation_mask(mean, destination / "maska_nasicenja_adc.png",
                                 args.adc_max, args.saturation_margin,
                                 show_title=not args.no_title)
        if args.timeseries:
            make_timeseries(npz, destination / "casovni_potek_sosednjih_takslov.png",
                            args.duration_sec, show_title=not args.no_title)
    print(f"\nFigures available in: {destination}")


if __name__ == "__main__":
    main()
