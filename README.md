# DriveSense

Machine learning for road type classification, road condition classification,
and road quality scoring from vehicle-mounted acceleration measurements.
Developed by Vedant Aggarwal during research at the University of Wisconsin–Madison.

The broader project uses a Raspberry Pi with DAQ, GPS, and IMU hardware. This
repository contains the supplied offline analysis scripts. Data acquisition
and preprocessing software were not supplied and are not included.

## Methods

Signals are segmented into nonoverlapping 60-second windows at 836 Hz. Features
combine time-domain statistics, FFT frequency-band power fractions, and Mexican
hat continuous wavelet transform summaries. Condition and quality workflows
also include zero crossing rate and histogram entropy.

| Script | Task | Model / inputs |
| --- | --- | --- |
| `type.py` | Highway, main road, neighborhood classification | RBF SVM; Z acceleration and XY magnitude |
| `KNN.py` | Smooth, grooves, bumpy classification | Tuned KNN; XY magnitude and Z acceleration |
| `prediction.py` | Manually labeled road quality scoring | Random forest regression; Z acceleration |
| `graph.py` | Raw sensor visualization | Plots the final CSV column |

## Install and run

Use Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
On macOS/Linux, use `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python type.py
python KNN.py
python prediction.py
python graph.py
```

Add the original CSVs first, following [DATA_FORMAT.md](DATA_FORMAT.md).
No recordings or real-data results are bundled. The original manually assigned
quality scores remain in `prediction.py`; they require their matching recordings.
Models train when their script runs, so saved model weights are not required.
Plots appear interactively; save verified figures in `results/` for your README.

## Evaluation and cleanup

- KNN reserves 20% of original segments for validation before any preprocessing
  or oversampling. Imputation, scaling, oversampling, and neighbor tuning occur
  inside training folds through an imbalanced-learn pipeline.
- This random segment split does not establish generalization to unseen drives.
  For that claim, provide recording IDs and split by drive/session. Nearby
  segments can share conditions and measurement artifacts.
- SVM prints a training report and evaluates the supplied separate test files.
  Use independent recordings and avoid duplicate data between the folders.
- Quality R² uses continuous predictions paired by original filename and segment
  ID. Rounded scores are displayed only for readability.
- Feature lengths are consistent: 30 for road type, 34 for condition, 17 for
  quality. Constant signals retain valid mean and range statistics.
- Histogram entropy uses normalized bin counts. Nonfinite segments are skipped
  without deleting samples and shifting the time axis.
- Importing the modules does not read recordings or train models.

These corrections can change earlier metrics. No accuracy or R² is claimed for
this cleaned version until it is rerun on the actual recordings. Dependencies
are unpinned; record tested versions for a reproducible release.

## Checks

```bash
python -m unittest discover -s tests -v
```

Checks cover feature shape and finite values, missing-data rejection, train-only
preprocessing/resampling, preserved segment/score alignment, and safe imports.
Synthetic test inputs verify behavior and are not research results.

The fixed scripts and support modules are intended to be uploaded together.
Raw CSVs are excluded from commits by default. Dataset access, sensor units,
manual score definitions, and collection details should be documented when the
original recordings are available.
