# MLflow Tracking & Leakage-Safe Validation

This document describes the validation + experiment-tracking layer added on top of
the original pipeline. It supersedes the ad-hoc single-split evaluation in
`src/Final_model.py` for **measuring** model quality. `Final_model.py` is kept for
its Folium map / GeoJSON export.

## Why this was needed

The original flow fit preprocessing on the whole dataset before splitting:

```
Data_preprocessing.py   ── fit imputer/PCA/scaler on ALL rows ──►  ML_ready.csv
Final_model.py          ── train_test_split(ML_ready.csv)      ──►  metrics
```

That leaks test-set statistics into training. Combined with a **random** split over
a 30 m grid whose neighbouring cells are near-duplicates, the reported
85 % accuracy / 92 % recall / 0.88 AUC were optimistic.

## What replaces it

| File | Role |
|------|------|
| `config.yaml` | Single source of truth: paths, features, CV settings, hyperparameters. |
| `src/flood_pipeline.py` | Leakage-safe `imblearn` Pipeline. Every stateful step (impute → rainfall PCA → scale → SMOTE) is re-fit inside each CV fold. Stateless feature engineering has a no-op `fit`. |
| `src/train.py` | Runs the validation schemes, selects the threshold on out-of-fold probabilities, calibrates, evaluates the cascade, and logs everything to MLflow (incl. Model Registry). |
| `requirements.txt` | Pinned dependencies. |

## How to run

```bash
pip install -r requirements.txt
python src/train.py --config config.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db     # open http://127.0.0.1:5000
```

One parent run (`hierarchical-pipeline`) with three nested runs: `detector`,
`severity`, `cascade`. Two models are registered: `flood-detector`, `flood-severity`.

## Validation techniques applied

1. **Leakage-safe pipeline** — impute/PCA/scale/SMOTE fit per fold, not globally.
2. **RepeatedStratifiedKFold** (5×3) — stable random-CV mean ± std.
3. **Spatial GroupKFold** — whole ~500 m UTM blocks held out, defeating
   neighbouring-cell autocorrelation. The random-vs-spatial gap = spatial optimism.
4. **Out-of-fold threshold selection** — the recall-target threshold is chosen on
   OOF probabilities, not a slice of the training data.
5. **Isotonic calibration** — makes the probabilities (and therefore the operating
   threshold) meaningful; tracked via Brier score.
6. **End-to-end cascade evaluation** — severity scored through the detector's
   decisions, so missed floods count as errors.

## Results (honest, cross-validated)

Detector:

| Metric | Random CV (5×3) | Spatial CV | README claim |
|--------|-----------------|-----------|--------------|
| ROC-AUC | 0.708 ± 0.034 | **0.565 ± 0.058** | 0.88 |
| Avg precision (PR-AUC) | 0.721 ± 0.036 | 0.577 ± 0.083 | — |
| Recall @0.5 | 0.667 ± 0.052 | 0.581 ± 0.214 | — |
| F1 @0.5 | 0.658 ± 0.029 | 0.552 ± 0.137 | 0.85 |

At the tuned threshold **0.326** (recall target 0.90): OOF recall **0.904**,
precision **0.554**, F1 0.687, accuracy 0.582. Isotonic calibration improved
Brier 0.240 → 0.220.

Severity (floods only): random-CV F1 0.513 / AUC 0.631; spatial-CV F1 0.436 /
**AUC 0.515** (≈ chance across space).

Cascade: missed-flood rate 0.096; 3-class accuracy on true floods 0.538.

### Interpretation

- The 90 % recall **is** reachable — but only by trading precision down to ~0.55
  and accuracy to ~0.58. The README's simultaneous 92 % / 85 % / 78 % came from the
  leaky, non-spatial evaluation.
- The **~0.14 ROC-AUC drop** from random to spatial CV is the real headline: most of
  the apparent skill was memorising neighbouring cells. Spatial AUC ~0.57 (detector)
  and ~0.51 (severity) mean the model barely generalises to unseen locations.
- **Action items:** the severity stage needs the most work (near-chance spatially);
  more flood events / years, spatial features, and hyperparameter search (Optuna,
  nested in the inner CV loop) are the obvious next levers — all now measurable and
  comparable in MLflow.

## Notes

- MLflow ≥ 3 rejects the legacy file store; this project uses `sqlite:///mlflow.db`.
- Models are logged with `cloudpickle` (the pipeline contains custom transformers and
  an XGBoost Booster), so `flood_pipeline.py` must be importable to load them.
- XGBoost is invariant to monotonic feature transforms, so the scaling/log/power
  steps barely affect it — the leakage that materially mattered here was **PCA,
  imputation, SMOTE, and threshold selection**.
