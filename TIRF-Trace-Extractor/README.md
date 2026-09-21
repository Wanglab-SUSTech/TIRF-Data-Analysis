# TIRF Trace Extractor

Extract 532/638 fluorescence intensity traces from ND2 recordings of two-color single-molecule colocalization experiments.

Author: **Zhihang Li**.

## Install and run

On Windows with Python 3.12 installed, open PowerShell in this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B main.py
```

After installation, you can also double-click `run_extract.bat`. A GPU is not required.

## Workflow and output

1. Load an ND2 file and check the 532/638 channel assignments and frame timing.
2. Select the detection channel and set the spot-detection and extraction parameters.
3. Review the detected ROIs and channel registration; adjust background subtraction and drift correction as needed.
4. Extract the traces and export them as CSV.

The CSV contains `Time_sec`, `ROI_ID`, `Net_532` and `Net_638`. The two `Net` columns contain background-subtracted intensities, including any channel scaling selected during extraction.

Use the tools in **Colocalization-Classifier** to screen the exported traces, assign them to classes and extract event dwell times.

## License and citation

Copyright (C) 2026 Zhihang Li. Licensed under the GNU General Public License,
version 3 only ([GPL-3.0-only](LICENSE)). See [CITATION.cff](CITATION.cff) for citation details.
