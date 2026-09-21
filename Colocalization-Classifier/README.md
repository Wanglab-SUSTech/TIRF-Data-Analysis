# Colocalization Classifier

Screen and manually classify two-color (532/638) fluorescence traces, identify events and extract their dwell times.

Author: **Zhihang Li**.

## Install and run

On Windows with Python 3.12 installed, open PowerShell in this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B signal_filter/run.py
```

After filtering and exporting, launch the classifier:

```powershell
.\.venv\Scripts\python.exe -B classifier/colocal_classifier.py
```

You can also use `run_signal_filter.bat` and `run_classifier.bat` after installation.
The filter requires tkinter/Tcl-Tk support in Python.

## Workflow and output

1. Open the CSV from **TIRF-Trace-Extractor** in `signal_filter`. Set the leakage corrections, intensity thresholds and minimum frame counts.
2. Review the traces and export the retained ROIs to `_filtered.csv`.
3. Open the filtered CSV in the classifier using **Load CSV...**. Assign traces to classes and adjust the event-detection settings.
4. Review event boundaries and export the selected classes. 

The input CSV must contain `Time_sec`, `ROI_ID`, `Net_532` and `Net_638`. The filter exports both `Net_*` and `Corrected_*` columns. Original intensities are stored in `Net_*` values; leakage correction is stored in `Corrected_*`. Filtering uses the corrected intensities, while the classifier reads `Net_532` and `Net_638`.

The classifier can export trace data (`_data.csv`), event summaries (`_events.csv`), event signals (`_event_signal.csv`) and analysis settings (`_params.txt`). In the event summary, `DwellTime_s` is the time of the last sample in an event minus the time of its first sample, in seconds.

## License and citation

Copyright (C) 2026 Zhihang Li. Licensed under the GNU General Public License, version 3 only ([GPL-3.0-only](LICENSE)). See [CITATION.cff](CITATION.cff) for citation details.
