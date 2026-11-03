
# Tactile sensor readout

Acquisition, visualization, and analysis software for a **12 × 32 flexible
piezoresistive tactile sensor** with 384 sensing elements (taxels). A custom
STM32 readout system streams tactile measurements over USB serial at
approximately **1 kHz**, using packed 12-bit ADC samples and CRC-16 validation.
The Python software provides real-time contact visualization, measurement
recording, and offline analysis.

This repository contains the host-side Python software. STM32 firmware is not
included.

## Current software features

- High-rate tactile acquisition at **1 kHz**, experimentally validated.
- Real-time heatmaps with baseline subtraction, snapshots, and frame diagnostics.
- Noise, drift, saturation, and contact-signal measurements in ADC counts.
- Repeated mold experiments with baseline, contact, and recovery recordings.
- Offline IoU/Dice evaluation, threshold sensitivity analysis, and thesis figures.
- Automated protocol and acquisition tests with simulated serial communication,
  including recovery from corrupted packets and timeout handling.

## Repository structure

```text
.
├── read_tactile_threaded.py          # Supported live viewer
├── check_tactile_noise.py            # Supported measurement receiver
├── tactile_mold_experiment.py        # Supported contact experiments
├── legacy/                           # Archived original 100 Hz software
│   ├── README.md
│   ├── read_tactile_threaded_100Hz.py
│   └── check_tactile_noise_100Hz.py
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
├── LICENSE                          # MIT license
└── .github/workflows/tests.yml
```

## Installation

Tested with Python 3.14.0. Compatibility with other Python versions has not
yet been verified. Run commands from the repository root:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
```

For runtime dependencies only, install `requirements.txt` instead. Dependencies
include NumPy, pyserial, OpenCV, Matplotlib, and Pillow.

## Communication protocol and acquisition frequency

Both protocols begin with `AA 55` and a 32-bit frame counter.

| Implementation | ADC payload | Integrity check | Frame size |
| --- | --- | --- | --- |
| Current, supported | 384 packed 12-bit samples | CRC-16/CCITT-FALSE over counter and payload | 584 bytes |
| Original, archived | 384 little-endian uint16 samples | No CRC | 774 bytes |

The current protocol packs two 12-bit ADC samples into three bytes, reducing
the frame size from 774 to 584 bytes while adding CRC-16 error detection.
Together with optimized STM32 firmware, this supports acquisition at
approximately 1 kHz.

The root-level `read_tactile_threaded.py`, `check_tactile_noise.py`, and
`tactile_mold_experiment.py` are the current supported implementation for the
584-byte packed-12-bit + CRC protocol. [`legacy/`](legacy/README.md) preserves
earlier software for historical reference and reproducibility. Its original
774-byte uint16/no-CRC protocol requires the matching older STM32 firmware.

**Acquisition frequency is controlled by the STM32 firmware, not by the Python scripts.** The current firmware operates at approximately 1 kHz using the 584-byte packed-12-bit protocol. The archived 100 Hz scripts require the original STM32 firmware and are incompatible with the current implementation.

### Acquisition performance

A continuous 180-second no-contact measurement with the current firmware
produced the following results:

| Metric | Measured result |
| --- | ---: |
| Recording duration | 180 s |
| Valid frames captured | 179,995 |
| Valid frame rate | **999.97 Hz** |
| Frame-counter discontinuities | 0 |
| CRC and packet-format errors | 0 |
| Serial timeouts | 0 |
| Host valid frames dropped | 0 |

The receiver recorded all expected transmitted frames within the captured
counter range, with no detected communication errors or frame losses.
The STM32 frame counter tracks completed scans passed to the transmission
function and cannot independently detect failed sensor scans.

The measurement receiver records incoming frames for analysis, while the
real-time viewer prioritizes the newest available frame for responsive
visualization. GUI refresh rate is therefore lower than the sensor acquisition
rate.

These results were obtained from a physical sensor measurement and describe
performance under the tested conditions. The automated tests use simulated
serial communication and do not independently verify hardware performance.

## Current sensor usage

Replace `PORT` with the device's serial port, such as `/dev/ttyACM0`,
`/dev/cu.usbmodem…`, or `COM3`.

```sh
python read_tactile_threaded.py --port PORT
python check_tactile_noise.py --port PORT --skip_touch --out_root thesis_noise

# Example contact experiment
python tactile_mold_experiment.py --port PORT \
  --mold_id three_stripe --load_g 200 --repeats 1
```

Keep the sensor unloaded while the viewer captures its baseline. Noise commands
above record a no-touch baseline; omit `--skip_touch` to include contact recording.
Mold experiments prompt for each baseline, contact, and recovery state. Supply
the actual mold identifier and applied mass; `--load_g` records grams, not
calibrated force. Capture windows describe steady states, not load-onset timing.

The viewer retains the newest frame for responsive display; the measurement
receiver retains frames for analysis and reports acquisition diagnostics.
Snapshots default to `tactile_snapshots_iou/`, and mold recordings to
`thesis_experiments/`.

## Offline analysis

Replace `SESSION`, `RUN`, and reference paths with existing local inputs.
Trial analysis accepts an extracted session directory or a ZIP archive.

```sh
python analysis/calculate_iou_dice.py --root tactile_snapshots_iou \
  --reference-mask /path/to/reference.npy --heatmap-top 0

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

The snapshot example uses `--heatmap-top 0` for the current viewer's default
headerless snapshots. Snapshots saved with `--show_header` have a 40-pixel header
and require `--heatmap-top 40`. Check historical snapshots for their actual format
before analysis; do not assume that a collection uses one format.

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

Raw experimental recordings and generated figures are excluded.

## License

Licensed under the [MIT License](LICENSE).
