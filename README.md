# Tactile sensor readout

Acquisition, visualization, and analysis software for a **12 × 32 flexible
piezoresistive tactile sensor** with 384 sensing elements (taxels). A custom
STM32 readout system streams ADC samples and frame counters to the host over
USB serial. The software displays contact patterns and records measurements
for sensor characterization and contact experiments.

This repository contains the host-side Python software. STM32 firmware is not
included.

## Features

- Real-time heatmaps with baseline subtraction, snapshots, and frame diagnostics.
- Noise, drift, saturation, and contact-signal measurements in ADC counts.
- Repeated mold experiments with baseline, contact, and recovery recordings.
- Offline IoU/Dice evaluation, threshold sensitivity analysis, and thesis figures.

## Repository structure

```text
.
├── read_tactile_threaded.py          # Current live viewer
├── read_tactile_threaded_100Hz.py     # Legacy live viewer
├── check_tactile_noise.py            # Current measurement receiver
├── check_tactile_noise_100Hz.py       # Legacy measurement receiver
├── tactile_mold_experiment.py        # Repeated contact experiments
├── analysis/
│   ├── calculate_iou_dice.py         # Snapshot-based evaluation
│   ├── calculate_iou_dice_thesis.py  # Numerical trial evaluation
│   ├── analyze_threshold_sensitivity.py
│   ├── plot_tactile_results.py
│   └── plot_contact_thesis.py
├── tests/
│   ├── test_tactile_protocol.py
│   └── test_tactile_acquisition.py
├── requirements.txt                 # Runtime dependencies
├── requirements-dev.txt             # Runtime + testing dependencies
└── .github/workflows/tests.yml
```

## Installation

Tested with Python 3.14.0. Compatibility with other Python versions has not yet been verified. Run commands from
the repository root:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
```

For runtime dependencies only, install `requirements.txt` instead. Dependencies
include NumPy, pyserial, OpenCV, Matplotlib, and Pillow.

## Firmware compatibility

Both protocols begin with `AA 55` and a 32-bit frame counter.

| Profile | ADC payload | Integrity check | Frame size |
| --- | --- | --- | --- |
| Current | 384 packed 12-bit samples | CRC-16/CCITT-FALSE over counter and payload | 584 bytes |
| Legacy 100 Hz | 384 little-endian uint16 samples | No CRC | 774 bytes |

Use the script matching the device firmware. The `_100Hz.py` scripts support
the legacy profile; Python does not configure the firmware frame rate.

## Sensor usage

Replace `PORT` with the device's serial port, such as `/dev/ttyACM0`,
`/dev/cu.usbmodem…`, or `COM3`.

```sh
# Current firmware
python read_tactile_threaded.py --port PORT
python check_tactile_noise.py --port PORT --skip_touch --out_root thesis_noise

# Legacy firmware
python read_tactile_threaded_100Hz.py --port PORT
python check_tactile_noise_100Hz.py --port PORT --skip_touch --out_root thesis_noise

# Example contact experiment using current firmware
python tactile_mold_experiment.py --port PORT \
  --mold_id three_stripe --load_g 200 --repeats 1
```

Keep the sensor unloaded while the viewer captures its baseline. Noise commands
above record a no-touch baseline; omit `--skip_touch` to include contact recording.
Mold experiments prompt for each baseline, contact, and recovery state. Supply
the actual mold identifier and applied mass; `--load_g` records grams, not
calibrated force. Capture windows describe steady states, not load-onset timing.

Viewers retain the newest frame for responsive display; measurement receivers
retain frames for analysis and report acquisition diagnostics. Snapshots default
to `tactile_snapshots_iou/`, and mold recordings to `thesis_experiments/`.

## Offline analysis

Replace `SESSION`, `RUN`, and reference paths with existing local inputs.
Trial analysis accepts an extracted session directory or a ZIP archive.

```sh
python analysis/calculate_iou_dice.py --root tactile_snapshots_iou \
  --reference-mask /path/to/reference.npy

python analysis/calculate_iou_dice_thesis.py \
  --data thesis_experiments/SESSION --reference /path/to/reference.npy \
  --out thesis_experiments/SESSION_reference_results

python analysis/analyze_threshold_sensitivity.py \
  --data thesis_experiments/SESSION --reference /path/to/reference.npy \
  --out thesis_experiments/SESSION_threshold_analysis

python analysis/plot_tactile_results.py thesis_noise/RUN/noise_results.npz \
  --all --timeseries

python analysis/plot_contact_thesis.py --results thesis_experiments/SESSION \
  --reference /path/to/reference.png --trial 1 \
  --output thesis_experiments/figures/contact_comparison
```

Choose a recorded trial for `--trial`. Relative input and output paths resolve
from the working directory; the snapshot evaluator's default root remains the
repository's `tactile_snapshots_iou/`. Run any script with `--help` for its options.

## Automated tests

```sh
python -m pytest -q tests/
```

The suite contains **42 tests and 8 subtests** covering packet decoding, CRC,
orientation, corruption recovery, buffering, frame counters, capture boundaries,
deadlines, and receiver shutdown. Tests use simulated serial input and need no
STM32 hardware, serial port, private data, or graphical desktop. They do not
validate physical sensor performance.

[GitHub Actions](.github/workflows/tests.yml) runs on pushes, pull requests, and
manual dispatch using Ubuntu 24.04 and Python 3.14. It installs development
dependencies, runs `python -m pip check`, and executes the test command above,
with read-only repository permissions and a 10-minute timeout.

## Experimental data

Experimental recordings, generated figures, and original reference data are not
distributed. Reproducing reported metrics requires the corresponding recordings
and independently aligned reference mask. The earlier evaluation used a
66-taxel reference; the current built-in geometric default produces 60 taxels
and is not an equivalent replacement.
