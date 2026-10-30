# Tactile sensor readout

Python software for acquiring and displaying a 12×32 tactile sensor over USB
serial, measuring noise and contact response, and analyzing saved thesis trials.
Acquisition, visualization, and mold experiment scripts remain in the repository
root; offline analysis scripts live in `analysis/`. Run the commands below from
the repository root.

## Existing environment and dependencies

The local baseline on 2026-10-30 uses Python 3.14.0 and pytest 9.2.0.
All 13 installed runtime dependency versions match [requirements.txt](requirements.txt),
which retains its existing exact pins. [requirements-dev.txt](requirements-dev.txt)
includes those runtime requirements and pins pytest and its test-only dependencies.
A fresh installation on another platform has not been verified.

| Distribution | Import | Used for |
| --- | --- | --- |
| `numpy` | `numpy` | Arrays, decoding, and numerical analysis |
| `pyserial` | `serial` | Sensor USB serial connection |
| `opencv-python` | `cv2` | Live viewer, snapshots, and some reference-image loading |
| `matplotlib` | `matplotlib` | Measurement and thesis figures |
| `pillow` | `PIL` | Reference images and snapshot analysis |

To create an equivalent environment, run from the repository root:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

## Import paths and tests

Keep the root scripts together. `tests/test_tactile_protocol.py` imports
`read_tactile_threaded`; `tests/test_tactile_acquisition.py` imports
`check_tactile_noise`. The mold experiment also imports `check_tactile_noise`
for `TactilePacketReader`, `ContinuousReceiver`, `discard_fresh_frames`,
`capture_frames`, and `json_safe`. The viewer imports `cv2` even during
hardware-free protocol testing.

With the environment activated, run these commands from the repository root so
the root modules remain importable:

```sh
python -m pip check
python -m pytest -q tests/
python -m pytest -v tests/test_tactile_protocol.py
```

The baseline suite passes all 42 tests and 8 subtests using simulated serial
input. It does not require a sensor or verify hardware operation.

The [Tests workflow](.github/workflows/tests.yml) runs on pushes, pull requests,
and manual dispatch from GitHub Actions. It uses Ubuntu and Python 3.14 to install
`requirements-dev.txt`, run `python -m pip check`, and execute
`python -m pytest -q tests/`. The job has read-only repository permissions and a
10-minute timeout. Tests run without a graphical desktop, STM32 hardware, a serial
port, or private measurement data.

## Sensor commands

Replace `PORT` with the connected device's serial port. A matching STM32 firmware
profile is required; the Python scripts do not configure its frame rate.

| Firmware profile | Packet | Viewer | Noise measurement |
| --- | --- | --- | --- |
| Current | Packed 12-bit samples with CRC16-CCITT-FALSE; 584 bytes | `read_tactile_threaded.py` | `check_tactile_noise.py` |
| Older 100 Hz | Little-endian uint16 samples without CRC; 774 bytes | `read_tactile_threaded_100Hz.py` | `check_tactile_noise_100Hz.py` |

```sh
python read_tactile_threaded.py --port PORT
python read_tactile_threaded_100Hz.py --port PORT

python check_tactile_noise.py --port PORT --skip_touch --out_root thesis_noise
python check_tactile_noise_100Hz.py --port PORT --skip_touch --out_root thesis_noise

python tactile_mold_experiment.py --port PORT --mold_id three_stripe --load_g 200
```

The mold identifier and 200 g load above are example metadata; supply the actual
mold and applied mass. The experiment uses the current packed12+CRC profile and
prompts for baseline, contact, and recovery states. Noise scripts also prompt
before capture; omit `--skip_touch` to include contact measurement.

Viewers retain the newest frame for live display. Their snapshots default to
`tactile_snapshots_iou/`; mold recordings default to `thesis_experiments/`.
Relative output paths resolve from the working directory. Run any script with
`--help` to inspect its existing options and defaults.

## Offline analysis commands

Use existing local recordings for `SESSION` and `RUN`, and supply your independent
reference mask where indicated. Trial analysis accepts either an extracted
session directory or a ZIP archive. These commands do not acquire new sensor data.

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
  --reference /path/to/reference.png --trial 4 \
  --output thesis_experiments/figures/contact_comparison
```

`calculate_iou_dice.py` still defaults to the repository-root
`tactile_snapshots_iou/`, independently of the working directory. Other relative
input and output paths retain their working-directory behavior. No measurement
directories or existing figures have moved. Use, for example,
`python analysis/plot_contact_thesis.py --help` to inspect a script's options.

## Local data and public-release review

Measurement directories, numerical recordings, generated reports and figures,
archives, caches, and local configuration are excluded by `.gitignore`.
Python source, tests, README files, and `requirements*.txt` files remain eligible
for tracking. Custom output directories should be placed under an ignored output
directory, particularly when producing JSON or other unrestricted file types.
Ignore rules do not remove files already tracked or erase Git history.

The reference mask used for the earlier IoU/Dice evaluation (`stripe_mask_4mm_binary_25pct.npy`, 66 active taxels) is retained separately and is not distributed with this repository. Analysis scripts accept external reference masks through command-line arguments. Reproducing the original results requires the corresponding reference mask; the current generated default differs from it.
