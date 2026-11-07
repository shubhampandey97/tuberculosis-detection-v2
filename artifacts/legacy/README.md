# Legacy results: INVALID, do not report

These metrics and plots come from the original pipeline (commit `050effd`). They are kept **only** for a before/after comparison (goal 3). They are not valid results:

- Training scaled images to [0,1], but evaluation used raw 0–255. No model got its own `preprocess_input` (docs/audit.md §5.1).
- The "test" set was the validation split used during training. There was no held-out test set (§5.3).
- The split was not stratified, and duplicate images could appear on both sides (§5.3).

The current results are in `artifacts/metrics/`, evaluated on the held-out test split from `data/splits.csv`.
