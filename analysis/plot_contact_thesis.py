#!/usr/bin/env python3
"""Make a four-panel thesis figure from an existing tactile contact experiment.

Use stored delta values and a saved 12x32 reference PNG; no new acquisition.

Example from the repository root:
    python3 analysis/plot_contact_thesis.py \
        --results 'session.zip' \
        --reference 'reference_mask.png' \
        --trial 4 --threshold 500 \
        --output 'primerjava_kontaktnega_odziva'
"""
import argparse
import io
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image


def load_reference(path: Path) -> np.ndarray:
    # White cell centres encode active taxels in a uniformly scaled 12x32 grid.
    rgb = np.asarray(Image.open(path).convert('RGB'))
    height, width = rgb.shape[:2]
    if height % 12 or width % 32 or height // 12 != width // 32:
        raise ValueError('Reference PNG must consist of a uniformly scaled 12 x 32 taxel grid.')
    pitch = height // 12
    centers = rgb[pitch // 2::pitch, pitch // 2::pitch]
    if centers.shape != (12, 32, 3):
        raise ValueError('Reference grid shape mismatch.')
    mask = np.all(centers > 200, axis=-1)
    return mask


def load_delta(results: Path, trial: int) -> np.ndarray:
    target = f'trial_{trial:03d}/measurements.npz'
    if results.is_dir():
        matches = list(results.rglob(target))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one {target} in directory {results}, got {len(matches)}')
        data = np.load(matches[0], allow_pickle=False)
    else:
        with ZipFile(results) as archive:
            matches = [name for name in archive.namelist() if name.endswith('/' + target)]
            if len(matches) != 1:
                raise FileNotFoundError(f'Expected one {target} inside ZIP, got {len(matches)}')
            data = np.load(io.BytesIO(archive.read(matches[0])), allow_pickle=False)
    with data:
        delta = np.asarray(data['delta'], dtype=np.float64)
    if delta.shape != (12, 32):
        raise ValueError(f'Expected 12x32 delta, got {delta.shape}')
    return delta


def configure_grid(ax, left=False):
    # Equal aspect preserves square taxels in both output formats.
    ax.set_xlim(0.5, 32.5)
    ax.set_ylim(12.5, 0.5)
    ax.set_aspect('equal')
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.set_yticks([1, 6, 12])
    ax.set_xlabel('Stolpec', labelpad=4)
    if left:
        ax.set_ylabel('Vrstica', labelpad=5)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=3, pad=3)


def plot(ref, delta, threshold, trial, output):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 16,
        'axes.titlesize': 17,
        'axes.labelsize': 17,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'pdf.fonttype': 42,
    })
    response = np.maximum(delta, 0)
    detected = delta >= threshold
    tp = int(np.count_nonzero(ref & detected))
    fp = int(np.count_nonzero(~ref & detected))
    fn = int(np.count_nonzero(ref & ~detected))
    iou = tp / (tp + fp + fn)
    dice = 2 * tp / (2 * tp + fp + fn)

    # 0=neither, 1=TP, 2=FP, 3=FN.
    comparison = np.zeros(ref.shape, dtype=np.uint8)
    comparison[ref & detected] = 1
    comparison[~ref & detected] = 2
    comparison[ref & ~detected] = 3

    fig = plt.figure(figsize=(12.8, 7.2), facecolor='white')
    positions = [
        (0.075, 0.57, 0.362, 0.30),
        (0.532, 0.57, 0.362, 0.30),
        (0.075, 0.20, 0.362, 0.30),
        (0.532, 0.20, 0.362, 0.30),
    ]
    axes = [fig.add_axes(p) for p in positions]
    ref_cmap = ListedColormap(['#101010', '#ffffff'])
    axes[0].imshow(ref.astype(np.int8), cmap=ref_cmap, vmin=0, vmax=1,
                   interpolation='nearest', origin='upper', extent=(0.5,32.5,12.5,0.5))
    axes[0].set_title(f'(a) Referenčna maska ({int(ref.sum())} takslov)', pad=9)
    configure_grid(axes[0], left=True)

    max_display = max(2200, int(np.ceil(response.max()/200)*200))
    img = axes[1].imshow(response, cmap='viridis', vmin=0, vmax=max_display,
                         interpolation='nearest', origin='upper', extent=(0.5,32.5,12.5,0.5))
    axes[1].set_title('(b) Izmerjeni odziv', pad=9)
    configure_grid(axes[1])
    cax = fig.add_axes([0.911, 0.598, 0.014, 0.244])
    cb = fig.colorbar(img, cax=cax)
    cb.set_label('Odziv [vrednosti ADC]', fontsize=15, labelpad=6)
    cb.ax.tick_params(labelsize=13)

    axes[2].imshow(detected.astype(np.int8), cmap=ref_cmap, vmin=0, vmax=1,
                   interpolation='nearest', origin='upper', extent=(0.5,32.5,12.5,0.5))
    axes[2].set_title(f'(c) Binarna maska (T = {threshold:g} ADC)', pad=9)
    configure_grid(axes[2], left=True)

    comparison_cmap = ListedColormap(['#101010', '#3fc45a', '#ed5149', '#4197e4'])
    comparison_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], comparison_cmap.N)
    axes[3].imshow(comparison, cmap=comparison_cmap, norm=comparison_norm,
                   interpolation='nearest', origin='upper', extent=(0.5,32.5,12.5,0.5))
    axes[3].set_title('(d) Ujemanje z referenco', pad=9)
    configure_grid(axes[3])
    legend_handles = [
        Patch(facecolor='#3fc45a', label=f'TP = {tp}'),
        Patch(facecolor='#ed5149', label=f'FP = {fp}'),
        Patch(facecolor='#4197e4', label=f'FN = {fn}'),
    ]
    # Place TP/FP/FN labels outside panel (d) to avoid obscuring taxels.
    axes[3].legend(handles=legend_handles, loc='center left',
                   bbox_to_anchor=(1.025, 0.5), ncol=1, frameon=False,
                   borderaxespad=0, labelspacing=0.8,
                   handlelength=1.15, handletextpad=0.6, fontsize=14)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f'Trial {trial}: ref={ref.sum()}, detected={detected.sum()}, TP={tp}, FP={fp}, FN={fn}, IoU={iou:.3f}, Dice={dice:.3f}')
    print('Saved:', output.with_suffix('.png'), 'and', output.with_suffix('.pdf'))


def main():
    ap = argparse.ArgumentParser(description='Plot reference, response, threshold and TP/FP/FN comparison.')
    ap.add_argument('--results', required=True, type=Path, help='Experiment ZIP or extracted experiment directory')
    ap.add_argument('--reference', required=True, type=Path, help='Saved 12x32 reference-mask PNG')
    ap.add_argument('--trial', type=int, default=4, help='Recorded trial number, default 4')
    ap.add_argument('--threshold', type=float, default=500.0)
    ap.add_argument('--output', type=Path, default=Path('primerjava_kontaktnega_odziva'))
    args = ap.parse_args()
    ref = load_reference(args.reference)
    delta = load_delta(args.results, args.trial)
    plot(ref, delta, args.threshold, args.trial, args.output)


if __name__ == '__main__':
    main()
