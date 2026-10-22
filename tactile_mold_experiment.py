#!/usr/bin/env python3
"""Record repeatable tactile mold measurements at full USB frame rate.

The file MUST be placed beside your already-tested check_tactile_noise.py.
It reuses that script's continuous CRC-checked 12-bit packet receiver rather
than the live viewer, which deliberately discards frames for low-lag display.

For each independent placement, record in order:
    1. unloaded baseline (mold removed)
    2. steady contact (mold and known load applied)
    3. unloaded recovery (mold and load removed)

User confirms each stable state by pressing ENTER. The USB receiver keeps
reading (discarding) packets during manual repositioning and settling; recorded
intervals are separate, time-defined host capture windows. Raw data and frame
counters are saved for every phase and repeated placement. No sample-accurate
load onset/offset is captured; use another protocol for response-time studies.

Examples:
    python3 tactile_mold_experiment.py --port /dev/cu.usbmodem2078388A45461 \\
      --mold_id three_stripe --load_g 200 --repeats 1 --baseline_sec 2 \\
      --contact_sec 2 --recovery_sec 2 --trial_name mold_pilot

    python3 tactile_mold_experiment.py --port /dev/cu.usbmodem2078388A45461 \\
      --mold_id three_stripe --load_g 200 --repeats 10 --trial_name mold_200g \\
      --baseline_sec 3 --contact_sec 3 --recovery_sec 3 \\
      --reference_mask reference_12x32.npy

Reference mask is optional and must be a correctly oriented 12x32 binary array.
Without it, the script saves the predicted mask but does not claim an IoU.
"""

import argparse
import csv
import json
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial

try:
    from check_tactile_noise import (
        TactilePacketReader,
        ContinuousReceiver,
        discard_fresh_frames,
        capture_frames,
        json_safe,
    )
except ImportError as exc:
    raise SystemExit(
        "Could not import check_tactile_noise.py. Place this file in the same "
        "folder as the tested check_tactile_noise.py and run it there. "
        f"Underlying error: {exc}"
    )

ROWS, COLS = 12, 32


def args_parser():
    p = argparse.ArgumentParser(description="Repeatable three-phase tactile mold measurements")
    p.add_argument('--port', required=True, help='Serial port, e.g. /dev/cu.usbmodem...')
    p.add_argument('--baud', type=int, default=115200, help='USB CDC setting (usually not throughput limiting)')
    p.add_argument('--serial_timeout', type=float, default=0.02)
    p.add_argument('--mold_id', required=True, help='Documented mold identifier')
    p.add_argument('--load_g', type=float, required=True, help='Mass added on mold in grams; not calibrated force')
    p.add_argument('--repeats', type=int, default=10, help='Number of independent mold placements')
    p.add_argument('--baseline_sec', type=float, default=3.0)
    p.add_argument('--contact_sec', type=float, default=3.0)
    p.add_argument('--recovery_sec', type=float, default=3.0)
    p.add_argument('--settle_sec', type=float, default=0.5, help='Additional settling after user confirms')
    p.add_argument('--warmup_frames', type=int, default=30)
    p.add_argument('--threshold', type=float, default=500.0,
                   help='Positive baseline-subtracted counts considered active')
    p.add_argument('--signal_vmax', type=float, default=2200.0,
                   help='Fixed upper display scale for signal heatmaps')
    p.add_argument('--reference_mask', default=None,
                   help='Optional aligned 12x32 binary .npy/.csv/.png mask')
    p.add_argument('--trial_name', default='mold_test')
    p.add_argument('--out_root', default='thesis_experiments')
    p.add_argument('--notes', default='', help='Rigid mount, orientation, mold details, etc.')
    a = p.parse_args()
    if a.repeats < 1 or a.warmup_frames < 1:
        p.error('repeats and warmup_frames must be positive')
    if min(a.baseline_sec, a.contact_sec, a.recovery_sec) <= 0 or a.settle_sec < 0:
        p.error('durations must be positive; settle_sec cannot be negative')
    if a.load_g < 0 or a.threshold < 0 or a.signal_vmax <= 0:
        p.error('load_g, threshold must be nonnegative; signal_vmax must be positive')
    return a


def load_reference(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Reference mask does not exist: {path}')
    if path.suffix.lower() == '.npy':
        arr = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.csv':
        arr = np.loadtxt(path, delimiter=',')
    elif path.suffix.lower() in ('.png', '.bmp', '.tif', '.tiff'):
        import cv2
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f'Could not load reference image: {path}')
    else:
        raise ValueError('Reference mask must be a .npy, .csv or grayscale image')
    if arr.shape != (ROWS, COLS):
        raise ValueError(f'Reference shape {arr.shape}; expected ({ROWS}, {COLS}). '
                         'Convert and align the DXF-derived reference to the sensor grid first.')
    if not np.all(np.isfinite(arr)):
        raise ValueError('Reference mask includes NaN or infinite values')
    return np.asarray(arr > 0, dtype=bool)


def summary_matrix_stats(arr):
    return {
        'min': float(np.min(arr)), 'median': float(np.median(arr)),
        'mean': float(np.mean(arr)), 'max': float(np.max(arr)),
    }


def calculate_results(baseline_frames, contact_frames, recovery_frames, threshold, reference=None):
    """Derive stable-plateau mean maps and thresholded positive contact region."""
    for label, frames in [('baseline',baseline_frames),('contact',contact_frames),('recovery',recovery_frames)]:
        if frames.ndim != 3 or frames.shape[1:] != (ROWS, COLS) or frames.shape[0] == 0:
            raise ValueError(f'{label} frames have unexpected shape {frames.shape}')
    baseline_mean = baseline_frames.mean(axis=0, dtype=np.float64)
    contact_mean = contact_frames.mean(axis=0, dtype=np.float64)
    recovery_mean = recovery_frames.mean(axis=0, dtype=np.float64)
    baseline_std = baseline_frames.std(axis=0, dtype=np.float64)
    delta = contact_mean - baseline_mean  # positive changes = activation
    recovery_delta = recovery_mean - baseline_mean
    active = delta >= threshold
    metrics = {
        'threshold_adc_counts': float(threshold),
        'active_taxels': int(active.sum()),
        'signal_positive_max_counts': float(max(delta.max(), 0.0)),
        'signal_signed_min_counts': float(delta.min()),
        'signal_signed_mean_counts': float(delta.mean()),
        'signal_mean_in_active_taxels_counts': float(delta[active].mean()) if active.any() else None,
        'baseline_median_temporal_std_counts': float(np.median(baseline_std)),
        'baseline_max_temporal_std_counts': float(np.max(baseline_std)),
        'recovery_median_absolute_change_counts': float(np.median(np.abs(recovery_delta))),
        'recovery_max_absolute_change_counts': float(np.max(np.abs(recovery_delta))),
        'recovery_mean_absolute_change_counts': float(np.mean(np.abs(recovery_delta))),
        'baseline_mean': summary_matrix_stats(baseline_mean),
        'contact_mean': summary_matrix_stats(contact_mean),
        'recovery_mean': summary_matrix_stats(recovery_mean),
        'near_high_saturation_contact_taxels': int(np.sum(contact_mean >= 4050)),
        'near_low_saturation_contact_taxels': int(np.sum(contact_mean <= 20)),
    }
    if reference is not None:
        intersection = int(np.logical_and(active, reference).sum())
        union = int(np.logical_or(active, reference).sum())
        total = int(active.sum() + reference.sum())
        metrics.update({
            'reference_taxels': int(reference.sum()),
            'intersection_taxels': intersection,
            'union_taxels': union,
            'iou': float(intersection / union) if union else None,
            'dice': float(2.0 * intersection / total) if total else None,
        })
    matrices = dict(baseline_mean=baseline_mean, contact_mean=contact_mean,
                    recovery_mean=recovery_mean, baseline_std=baseline_std,
                    delta=delta, recovery_delta=recovery_delta,
                    active_mask=active.astype(np.uint8))
    return matrices, metrics


def save_figure(path, arr, label, vmin, vmax):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        img = ax.imshow(arr, origin='upper', interpolation='nearest', aspect='auto',
                        vmin=vmin, vmax=vmax)
        fig.colorbar(img, ax=ax, label=label)
        ax.set_xlabel('Column (0–31)')
        ax.set_ylabel('Row (0–11)')
        ax.set_xticks(range(0, 32, 2))
        ax.set_yticks(range(0, 12, 1))
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
    except ImportError:
        print('[WARN] matplotlib not installed; numerical arrays saved, heatmap PNG omitted.')


def record_phase(receiver, name, seconds, settle):
    descriptions = {
        'baseline': 'REMOVE the mold and load; sensor completely unloaded and stable.',
        'contact': 'APPLY the mold and known load; hands off; load stable.',
        'recovery': 'REMOVE the mold and load completely; sensor unloaded and stable.',
    }
    input(f'  {name.upper()}: {descriptions[name]}\n  Press ENTER when ready...')
    # Reader continues draining during the physical setup and this settling delay.
    before = receiver.snapshot_stats()['valid_frames_discarded']
    if settle:
        time.sleep(settle)
    discarded = receiver.snapshot_stats()['valid_frames_discarded'] - before
    frames, fps, stats = capture_frames(receiver, seconds, discarded)
    print(f'    -> {len(frames)} frames / {stats["actual_duration_s"]:.3f} s '
          f'({fps:.2f} Hz), counter skips={stats["counter_skips"]}, '
          f'CRC errors={stats["crc_errors"]}')
    # ADC values are uint12; store exact values using uint16.
    if not np.all((frames >= 0) & (frames <= 4095)):
        raise ValueError('Unexpected ADC values outside 0..4095')
    return frames.astype(np.uint16), stats


def save_trial(directory, trial_number, measurements, stats, matrices, metrics, a, reference):
    directory.mkdir(parents=True, exist_ok=False)
    saved = {}
    for phase in ('baseline', 'contact', 'recovery'):
        frames, meta = measurements[phase], stats[phase]
        saved[f'{phase}_frames'] = frames
        saved[f'{phase}_counters'] = meta['counters']
    saved.update(matrices)
    if reference is not None:
        saved['reference_mask'] = reference.astype(np.uint8)
    np.savez_compressed(directory/'measurements.npz', **saved)
    metadata = {
        'trial_index':trial_number, 'recorded_at':datetime.now().isoformat(timespec='seconds'),
        'mold_id':a.mold_id, 'load_g':a.load_g, 'notes':a.notes,
        'sensor_shape':[ROWS,COLS], 'orientation':'rows x columns, left/right flipped to match viewer',
        'signal_definition':'mean ADC during stable contact minus mean unloaded ADC before contact',
        'threshold_rule':'signal >= threshold; signed positive response only',
        'timing_note':'Host-timed steady-state intervals, no per-frame MCU timestamps or actuation onset',
        'phases':{name:{k:v for k,v in json_safe(s).items() if k != 'counters'}
                  for name,s in stats.items()},
        'metrics': metrics,
    }
    (directory/'trial_summary.json').write_text(json.dumps(metadata,indent=2), encoding='utf-8')
    save_figure(directory/'signal_delta_fixed.png', matrices['delta'], 'ADC counts (contact – baseline)',
                -a.signal_vmax, a.signal_vmax)
    save_figure(directory/'positive_signal_fixed.png', np.maximum(matrices['delta'],0),
                'Positive ADC response', 0, a.signal_vmax)
    save_figure(directory/'binary_contact_mask.png', matrices['active_mask'],
                'Activated taxels (0 / 1)', 0, 1)
    print(f'  Saved: {directory}')


def write_session_csv(root, rows):
    if not rows:
        return
    cols = list(rows[0])
    with (root/'trial_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    a = args_parser()
    reference = load_reference(a.reference_mask)
    if reference is None:
        print('[INFO] No aligned 12x32 reference mask: IoU / Dice are deliberately not calculated.')
    if a.threshold != 500:
        print('[INFO] This run uses a different ADC threshold than the 500-count ERK value.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = re.sub(r'[^A-Za-z0-9_-]', '_', a.trial_name)
    root = Path(a.out_root)/f'{safe_name}_{stamp}'
    root.mkdir(parents=True,exist_ok=False)
    (root/'session_config.json').write_text(json.dumps({
        'started_at':datetime.now().isoformat(timespec='seconds'),
        'args':vars(a), 'python':sys.version.split()[0], 'platform':platform.platform(),
        'packet':'AA55 + uint32 counter + 576-byte packed12 payload + CRC16 CCITT-FALSE, 584 bytes',
        'notes':'All 384 taxels included. Contact onset and removal are NOT sample-aligned.',
    },indent=2), encoding='utf-8')
    if reference is not None:
        np.save(root/'reference_mask_12x32.npy', reference.astype(np.uint8))
    print(f'\n[INFO] Output directory: {root}\n'
          f'[INFO] {a.repeats} independent placements, {a.load_g:g} g, mold={a.mold_id}, '
          f'threshold={a.threshold:g} ADC counts')
    print('[INFO] Use a flat rigid support; keep the mold alignment and load consistent.')
    print('[INFO] Enter is pressed AFTER each physical state is established, not during motion.')
    ser = None
    receiver = None
    csv_rows = []
    try:
        ser = serial.Serial(a.port,a.baud,timeout=a.serial_timeout)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        receiver = ContinuousReceiver(TactilePacketReader(ser=ser,rows=ROWS,cols=COLS))
        receiver.start()
        discard_fresh_frames(receiver,a.warmup_frames,timeout_s=10.0)
        print('[INFO] Valid packed12+CRC packets received. Warmup complete.')
        for i in range(1,a.repeats+1):
            print(f'\n========== Placement {i}/{a.repeats} ==========')
            measurements = {}
            stats = {}
            for phase, seconds in (('baseline',a.baseline_sec),('contact',a.contact_sec),('recovery',a.recovery_sec)):
                measurements[phase],stats[phase] = record_phase(receiver,phase,seconds,a.settle_sec)
            matrices, metrics = calculate_results(
                measurements['baseline'], measurements['contact'], measurements['recovery'],
                threshold=a.threshold,reference=reference)
            save_trial(root/f'trial_{i:03d}', i, measurements,stats,matrices,metrics,a,reference)
            row = {
                'trial':i,'mold_id':a.mold_id,'load_g':a.load_g,'threshold_adc':a.threshold,
                'baseline_frames':len(measurements['baseline']),
                'contact_frames':len(measurements['contact']),
                'recovery_frames':len(measurements['recovery']),
                'baseline_fps':stats['baseline']['fps'],
                'contact_fps':stats['contact']['fps'],
                'recovery_fps':stats['recovery']['fps'],
                'baseline_counter_skips':stats['baseline']['counter_skips'],
                'contact_counter_skips':stats['contact']['counter_skips'],
                'recovery_counter_skips':stats['recovery']['counter_skips'],
                'crc_errors_total':sum(stats[k]['crc_errors'] for k in ('baseline','contact','recovery')),
                'active_taxels':metrics['active_taxels'],
                'max_positive_signal_adc':metrics['signal_positive_max_counts'],
                'baseline_median_std_adc':metrics['baseline_median_temporal_std_counts'],
                'recovery_median_abs_delta_adc':metrics['recovery_median_absolute_change_counts'],
                'iou':metrics.get('iou',''), 'dice':metrics.get('dice',''),
            }
            csv_rows.append(row)
            write_session_csv(root,csv_rows)
            print(f'  Activated taxels: {metrics["active_taxels"]}; '
                  f'max positive response: {metrics["signal_positive_max_counts"]:.1f} ADC')
            if reference is not None:
                print(f'  IoU: {metrics["iou"]}; Dice: {metrics["dice"]}')
            print(f'  Unloaded recovery, median |Δ|: '
                  f'{metrics["recovery_median_absolute_change_counts"]:.2f} ADC')
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted. Previously completed trials remain saved.')
    finally:
        if receiver is not None:
            receiver.stop()
        if ser is not None:
            ser.close()
    print(f'\n[INFO] Completed {len(csv_rows)} trials. Results: {root}')


if __name__=='__main__':
    main()
