# KymoTracker 3.4

A Tkinter + Matplotlib GUI tool for interactive kymograph tracking, designed for single-molecule fluorescence data from the C-Trap platform. Supports multi-channel display, multi-file batch processing, and automatic unit conversion.

## Features

- **Multi-channel kymograph display** — overlay multiple fluorescence channels; tracking parameters are saved per channel and auto-reloaded on switch
- **Interactive ROI & tracking** — draw rectangular or curved polygon ROIs; `Z` undoes the last vertex, `Enter` confirms, `Esc` cancels
- **Automatic unit conversion** — built-in nm ↔ bp conversion (1 bp = 0.34 nm)
- **Analysis**:
  - MSD (mean squared displacement) calculation
  - Diffusion / rate estimation
  - Segment-wise statistics
- **Multi-file support** — batch loading of `.h5` kymograph files via [lumicks.pylake](https://lumicks.github.io/pylake/)
- **Export**:
  - Excel (`.xlsx`, all traces in one multi-sheet workbook)
  - CSV (one file per trace + settings file + metadata file)
  - Standard and wide data layouts
- **Extras** — high-resolution kymograph PNG export (300 dpi), caching for fast redraws, track-point deletion with history

## Requirements

- Python 3.8
- Dependencies:

```bash
pip install pylake==0.8.1 numpy pandas matplotlib scikit-image openpyxl
```

> Note: ROI masking prefers scikit-image and automatically falls back to OpenCV if unavailable (optionally install `opencv-python`).

## Usage

```bash
python "KymoTracker3.4 .py"
```

1. Click **Load** to select C-Trap kymograph `.h5` files
2. Draw ROIs on the image and run tracking
3. Tune tracking parameters and switch channels to inspect multi-color signals
4. Click **Export** to save analysis results (xlsx / csv), or export the kymograph as a PNG

## Notes

- The script filename contains a space (`KymoTracker3.4 .py`) — quote it on the command line, or rename the file
- `.h5` files must be exported by a Lumicks C-Trap (read via pylake)

## License

For internal laboratory research use only.
