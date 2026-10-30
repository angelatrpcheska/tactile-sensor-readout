#!/usr/bin/env python3
"""Threshold sensitivity analysis of tactile mold experiment recordings.

Uses saved unthresholded mean ADC differences ("delta") and an independently
specified 12x32 binary reference mask. NEVER recalibrates the reference
alignment or chooses an "optimal" threshold on the evaluation trials.

Usage from tactile-sensor-readout/:
    python3 analysis/analyze_threshold_sensitivity.py \
      --data "thesis_experiments/three_stripe_200g_20261022_170808" \
      --reference "reference_mask.png" \
      --out "thesis_experiments/three_stripe_threshold_analysis"

Also accepts a ZIP file with the same trial_###/measurements.npz structure.
Requirements: numpy, matplotlib, pillow.
"""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROWS, COLS = 12, 32


def read_reference(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == '.npy':
        a = np.load(path, allow_pickle=False)
    elif ext == '.csv':
        a = np.loadtxt(path, delimiter=',')
    elif ext == '.png':
        a = np.asarray(Image.open(path).convert('L'), dtype=np.float64)
        h, w = a.shape
        if (h, w) != (ROWS, COLS):
            if h % ROWS or w % COLS or h // ROWS != w // COLS:
                raise ValueError('Reference PNG must be 12x32 or a clean uniform enlargement (no axes/title)')
            a = a.reshape(ROWS, h // ROWS, COLS, w // COLS).mean(axis=(1, 3))
        a = a > (0.5 if np.max(a) <= 1 else 127.5)
    else:
        raise ValueError('Reference must be .npy, .csv, or clean .png')
    a = np.squeeze(a)
    if a.shape != (ROWS, COLS):
        raise ValueError(f'Reference shape is {a.shape}, expected 12x32')
    if not np.all(np.isfinite(a)) or not np.isin(a, [0, 1]).all():
        raise ValueError('Expected a clean binary 0/1 reference mask')
    return a.astype(bool)


def iter_trials(path: Path):
    if path.is_dir():
        files = sorted(path.glob('trial_*/measurements.npz'))
        if not files:
            raise FileNotFoundError(f'No trial_*/measurements.npz under {path}')
        for file in files:
            with np.load(file, allow_pickle=False) as z:
                yield file.parent.name, np.array(z['delta'], dtype=np.float64), np.array(z['active_mask'], dtype=bool)
    elif path.is_file() and path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as zf:
            members = sorted(n for n in zf.namelist() if n.endswith('/measurements.npz') and '/trial_' in n)
            if not members:
                raise FileNotFoundError('No trial_*/measurements.npz in ZIP')
            for member in members:
                with np.load(io.BytesIO(zf.read(member)), allow_pickle=False) as z:
                    yield member.split('/')[-2], np.array(z['delta'], dtype=np.float64), np.array(z['active_mask'], dtype=bool)
    else:
        raise FileNotFoundError(f'Experiment folder or ZIP not found: {path}')


def save_csv(path: Path, rows: list[dict]):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True, help='Extracted experiment folder or ZIP')
    p.add_argument('--reference', type=Path, required=True, help='Original aligned 12x32 reference mask')
    p.add_argument('--out', type=Path, required=True, help='New output folder')
    p.add_argument('--figures-dir', type=Path, default=Path('thesis_experiments/figures'),
                   help='Thesis figure export folder (default: thesis_experiments/figures)')
    p.add_argument('--min-threshold', type=int, default=100)
    p.add_argument('--max-threshold', type=int, default=1200)
    p.add_argument('--step', type=int, default=50)
    p.add_argument('--primary-threshold', type=int, default=500)
    args = p.parse_args()
    if args.step <= 0 or args.max_threshold < args.min_threshold:
        p.error('Threshold interval or step is invalid')

    thresholds = sorted(set(range(args.min_threshold, args.max_threshold + 1, args.step)) | {args.primary_threshold})
    reference = read_reference(args.reference)
    trials = list(iter_trials(args.data))
    if not trials:
        raise ValueError('No experimental trials found')
    for name, delta, mask in trials:
        if delta.shape != (ROWS, COLS) or mask.shape != (ROWS, COLS):
            raise ValueError(f'{name}: expected 12x32 arrays')
        if not np.all(np.isfinite(delta)):
            raise ValueError(f'{name}: nonfinite ADC values')
        if not np.array_equal(mask, delta >= args.primary_threshold):
            raise ValueError(f'{name}: saved mask does not match primary threshold {args.primary_threshold}')

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Output folder is nonempty: {args.out}. Use a new folder to preserve past results.')
    args.out.mkdir(parents=True, exist_ok=True)

    per_trial = []
    summary = []
    for threshold in thresholds:
        metrics = []
        for name, delta, _ in trials:
            measured = delta >= threshold
            tp = int((reference & measured).sum())
            fp = int((~reference & measured).sum())
            fn = int((reference & ~measured).sum())
            den_iou, den_dice = tp + fp + fn, 2 * tp + fp + fn
            iou = tp / den_iou if den_iou else 1.0
            dice = 2 * tp / den_dice if den_dice else 1.0
            item = dict(threshold_adc=threshold, trial=name, active_taxels=int(measured.sum()),
                        TP=tp, FP=fp, FN=fn, IoU=iou, Dice=dice)
            per_trial.append(item)
            metrics.append(item)
        sd = lambda vals: float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        s = dict(threshold_adc=threshold, trials=len(trials), reference_taxels=int(reference.sum()),
                 mean_IoU=float(np.mean([r['IoU'] for r in metrics])),
                 sd_IoU=sd([r['IoU'] for r in metrics]),
                 mean_Dice=float(np.mean([r['Dice'] for r in metrics])),
                 sd_Dice=sd([r['Dice'] for r in metrics]),
                 mean_active_taxels=float(np.mean([r['active_taxels'] for r in metrics])),
                 sd_active_taxels=sd([r['active_taxels'] for r in metrics]),
                 mean_TP=float(np.mean([r['TP'] for r in metrics])),
                 mean_FP=float(np.mean([r['FP'] for r in metrics])),
                 mean_FN=float(np.mean([r['FN'] for r in metrics])))
        summary.append(s)

    save_csv(args.out / 'threshold_summary.csv', summary)
    save_csv(args.out / 'threshold_by_trial.csv', per_trial)
    np.save(args.out / 'reference_mask_used.npy', reference.astype(np.uint8))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.array([r['threshold_adc'] for r in summary])
    primary = next(r for r in summary if r['threshold_adc'] == args.primary_threshold)

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })
    with plt.rc_context({'text.usetex': False}):
        fig, ax = plt.subplots(figsize=(11, 4.6), layout='constrained', facecolor='white')
        ax.set_facecolor('white')
        for metric, sd, label, color in [
                ('mean_IoU', 'sd_IoU', 'IoU', '#1f77b4'),
                ('mean_Dice', 'sd_Dice', 'Dice', '#ff7f0e')]:
            mean_y = np.array([r[metric] for r in summary])
            sd_y = np.array([r[sd] for r in summary])
            ax.plot(x, mean_y, label=label, color=color, linewidth=1.0)
            ax.fill_between(x, mean_y - sd_y, mean_y + sd_y,
                            alpha=.13, color=color, linewidth=0)
        ax.axvline(args.primary_threshold, linestyle='--', linewidth=1.2,
                   color='0.5', label=f'Osnovni prag ({args.primary_threshold})')
        ax.set_xlabel('Prag zaznave (vrednosti ADC)')
        ax.set_ylabel('Koeficient')
        ax.set(xlim=(100, 1200), ylim=(0, 1))
        ax.set_xticks(np.arange(200, 1201, 200))
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.grid(alpha=.2)
        ax.legend(loc='best')
        # Retain the original analysis chart alongside the separate thesis exports.
        fig.savefig(args.out / 'threshold_iou_dice.png', dpi=300, bbox_inches='tight')
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figures_dir / 'vpliv_praga_zaznave.pdf')
        fig.savefig(args.figures_dir / 'vpliv_praga_zaznave.png', dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.6), layout='constrained', facecolor='white')
    active = np.array([r['mean_active_taxels'] for r in summary])
    sdactive = np.array([r['sd_active_taxels'] for r in summary])
    line, = ax.plot(x, active, linewidth=1.0, label='Zaznano kontaktno območje')
    ax.fill_between(x, active-sdactive, active+sdactive, alpha=.15, color=line.get_color())
    ax.axhline(int(reference.sum()), linestyle=':', color='gray', label='Referenčni taksli')
    ax.axvline(args.primary_threshold, linestyle='--', color='gray')
    ax.set(xlabel='Prag zaznave (vrednosti ADC)', ylabel='Število aktiviranih takslov')
    ax.grid(alpha=.2)
    ax.legend(loc='best')
    fig.savefig(args.out / 'threshold_active_taxels.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    info = {'experiment':str(args.data), 'reference':str(args.reference),
            'reference_active_taxels':int(reference.sum()), 'num_independent_placements':len(trials),
            'baseline_subtraction':'delta = mean(contact ADC) - mean(baseline ADC)',
            'primary_threshold_adc':args.primary_threshold,
            'primary_result':primary,
            'analysis_note':'Threshold sweep is descriptive on the same evaluation trials, not independent optimization of the threshold. Do not move or flip the reference separately for each trial.',
            'trials':[r[0] for r in trials]}
    (args.out/'analysis_summary.json').write_text(json.dumps(info, indent=2)+'\n',encoding='utf-8')
    best = max(summary, key=lambda r:r['mean_IoU'])
    print(f'Loaded {len(trials)} trials and reference mask with {int(reference.sum())} taxels.')
    print(f'At {args.primary_threshold} ADC: IoU={primary["mean_IoU"]:.4f} ± {primary["sd_IoU"]:.4f}, Dice={primary["mean_Dice"]:.4f} ± {primary["sd_Dice"]:.4f}')
    print(f'Highest observed mean IoU in the sweep: {best["mean_IoU"]:.4f} at {best["threshold_adc"]} ADC (descriptive, not validated optimum).')
    print('Saved CSVs, charts, and summary in:',args.out)

if __name__ == '__main__':
    main()
