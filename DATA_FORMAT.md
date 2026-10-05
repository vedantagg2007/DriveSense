# Data setup

Original recordings are not included in this repository. Place your CSV files
in the following locations beneath the project folder:

| Location | Used by |
| --- | --- |
| `combined/highway.csv` | KNN.py, graph.py |
| `combined/main road.csv` | KNN.py |
| `combined/neighborhood.csv` | KNN.py |
| `3 type/highway.csv` | type.py, prediction.py training |
| `3 type/main road.csv` | type.py, prediction.py training |
| `3 type/neighborhood.csv` | type.py, prediction.py training |
| `3 type/testing/highway.csv` | type.py, prediction.py testing |
| `3 type/testing/main road.csv` | type.py, prediction.py testing |
| `3 type/testing/neighborhood.csv` | type.py, prediction.py testing |

Alternatively, set `DRIVESENSE_DATA_DIR` to the parent directory containing
`combined` and `3 type`. Paths are resolved independently of the terminal's
working directory.

Every CSV requires a header row. The scripts use column positions:

- The last three columns must be numeric acceleration X, Y, and Z, in that order.
- The first column of the `combined` files must contain condition labels:
  `smooth`, `grooves`, or `bumpy`. Each segment receives its majority label.
- The first column of quality-test files is read for a descriptive label.
- Road-type labels come from the input file mapping, rather than a CSV column.
- Optional timestamps, GPS fields, and other metadata belong between the first
  label column and the last three acceleration columns.

For example, a minimal header is `condition,accel_x,accel_y,accel_z`.
Acceleration units must be consistent across recordings; the original files
were not supplied, so their physical units cannot be verified here.

Each nonoverlapping segment contains 50,160 samples, representing 60 seconds at
836 Hz. An incomplete final segment is discarded. Classification requires
multiple segments per class. Nonfinite acceleration values cause the relevant
segment to be skipped, without shifting later segment boundaries.

## Manually assigned quality scores

`prediction.py` preserves the supplied training score lists and 46 test scores.
They belong to the original recordings, ordered highway, main road, then
neighborhood, with one score per original 60-second segment. They are not labels
for an arbitrary replacement dataset. Update those lists and test mappings
together if using different recordings. Training uses only segments with an
assigned score. Test evaluation uses only segment IDs with a known score; other
predictions are displayed but excluded from R². Skipped segments retain their
original IDs so later scores remain aligned.

The `.gitignore` excludes CSV recordings by default. If you choose to share a
permitted sample dataset, add it explicitly with `git add -f path/to/sample.csv`
and explain its source. A schema alone does not enable model training.
