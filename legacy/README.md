# Archived original 100 Hz software

These scripts are historical backups of the original tactile sensor software,
retained for reference and reproducibility. They are not supported programs for
operating the current sensor.

The original STM32 firmware transmitted 384 ADC values as little-endian 16-bit
integers without CRC, in 774-byte packets. Reproducing that implementation
requires the matching older STM32 firmware.

Current firmware sends 384 packed 12-bit ADC values with CRC-16/CCITT-FALSE
validation in 584-byte packets. Running it at 100 Hz does not make this stream
compatible with the original uint16/no-CRC protocol. Acquisition frequency is
set in firmware; the `100Hz` filenames identify the original implementation,
not a frequency-selection option.

Use the current root-level `read_tactile_threaded.py`, `check_tactile_noise.py`,
and `tactile_mold_experiment.py` with current firmware. See the
[main README](../README.md) for supported usage.

The archived code, options, and example commands are retained unchanged for
historical reference. In the archived measurement script, `--expected_fps`
controls a rate warning; it does not set the acquisition frequency.
