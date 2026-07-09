# 🌊 Satellite-Based Flood Prediction Using Machine Learning

> **End-to-end geospatial ML pipeline for flood detection and severity classification from Sentinel-1 SAR imagery and digital elevation models, with leakage-safe cross-validation and MLflow experiment tracking.**
> Mohit Chauhan • Data Scientist / ML Engineer, IIT Roorkee

---

## 📋 Overview

A complete geospatial machine-learning pipeline that predicts **flood occurrence** and **severity** over a study area on the Tawi river (Jammu, India). It fuses Sentinel-1 SAR-derived flood labels with terrain/hydrology features from a Digital Elevation Model, and models the problem as a **two-stage hierarchical classifier**:

```
Stage A — Detector:  Flood?  (No-flood / Flood)
Stage B — Severity:  Given a flood, Low or High
```

The project's emphasis is **methodological rigor**: preprocessing is leakage-safe (fit inside every CV fold), evaluation uses **spatial** cross-validation appropriate to gridded geospatial data, and every run is tracked and versioned in MLflow.

### ✨ Key Features

- **Automated pipeline** — raw satellite/DEM data → engineered features → validated predictions.
- **SAR-based labeling** — no manual ground truth; flood masks from Sentinel-1 via Google Earth Engine.
- **Hierarchical classification** — detect floods, then classify severity only where a flood is predicted.
- **Leakage-safe** — imputation, PCA, scaling and SMOTE are fit per fold inside `scikit-learn`/`imbalanced-learn` Pipelines.
- **Spatial validation** — GroupKFold over spatial blocks, exposing autocorrelation optimism a random split hides.
- **Experiment tracking** — MLflow logging + Model Registry for reproducible, comparable, versioned models.
- **Recall-oriented operating point** — decision threshold tuned to **90% flood recall**, prioritizing the cost of a missed flood in disaster response.
- **Interactive outputs** — Folium risk map (HTML) and GeoJSON for GIS stakeholders.

---

## 🎯 Model Performance (honest, cross-validated)

Metrics come from `src/train.py` under two cross-validation schemes. The gap between **random** CV and **spatial** CV is the headline: it quantifies how much apparent skill was really neighbouring-cell autocorrelation.

**Flood Detector (binary)**

| Metric | Random CV (5×3) | Spatial CV (GroupKFold) |
|--------|-----------------|-------------------------|
| ROC-AUC | 0.708 ± 0.034 | **0.565 ± 0.058** |
| PR-AUC (avg precision) | 0.721 ± 0.036 | 0.577 ± 0.083 |
| F1 @0.5 | 0.658 ± 0.029 | 0.552 ± 0.137 |

At the tuned operating threshold **0.326** (recall target 0.90): **recall 0.904**, precision 0.554, F1 0.687, accuracy 0.582. Isotonic calibration improved Brier score **0.240 → 0.220 (8%)**.

**Severity Classifier (Low vs High, floods only)** — random-CV F1 0.513 / ROC-AUC 0.631; spatial-CV F1 0.436 / ROC-AUC 0.515.

**End-to-end cascade** — missed-flood rate 0.096; 3-class accuracy on true floods 0.538.

> ⚠️ **Note on earlier numbers.** A previous version of this pipeline reported ~85% accuracy / 92% recall / 0.88 AUC. Those came from preprocessing fit on the full dataset (data leakage) evaluated on a random split of spatially autocorrelated grid cells. After removing the leakage and validating spatially, the true generalization is materially lower — and now trustworthy. See [`docs/mlflow-and-validation.md`](docs/mlflow-and-validation.md).

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: KML/GeoJSON boundary · DEM raster · Sentinel-1 SAR · rainfall │
└──────────────────────────────────────────────────────────────┘
                             ↓
  STEP 1 — Feature extraction  (src/QGIS_DataExtraction_pipeline.py)
   • Clip DEM, fill depressions, D8 flow direction/accumulation
   • Terrain derivatives: slope, aspect, hillshade, TWI, SPI
   • 30 m sampling grid → raster sampling → feature table
                             ↓
  STEP 2 — SAR flood labeling  (src/Flood_label_extraction_GEE.js)
   • Sentinel-1 before/after water detection in Google Earth Engine
   • Labels: 0 = none · 1 = mild · 2 = severe
                             ↓
  STEP 3 — Leakage-safe preprocessing + training  (src/flood_pipeline.py, src/train.py)
   • imblearn Pipeline (impute → engineer → rainfall PCA → scale → SMOTE → XGBoost)
   • ALL stateful steps re-fit inside each CV fold
   • RepeatedStratifiedKFold + spatial GroupKFold
   • OOF threshold selection · isotonic calibration · cascade eval
   • MLflow tracking + Model Registry
                             ↓
  STEP 4 — Inference & visualization  (src/Final_model.py)
   • Predict over the grid → Folium risk map + GeoJSON + prediction CSV
```

---

## 🧰 Tech Stack

**ML / data:** Python, XGBoost, scikit-learn, imbalanced-learn (SMOTE), pandas, NumPy
**MLOps / validation:** MLflow (tracking + Model Registry), RepeatedStratifiedKFold, spatial GroupKFold, isotonic calibration, YAML-driven config
**Geospatial:** Google Earth Engine (Sentinel-1 SAR), GeoPandas, Shapely, QGIS/GDAL, Folium
**Viz:** Matplotlib · **Tooling:** Git, joblib

---

## 📁 Project Structure

```
Flood Prediction Model/
├── README.md
├── config.yaml                         # single source of truth (paths, features, CV, params)
├── requirements.txt
│
├── src/
│   ├── QGIS_DataExtraction_pipeline.py  # DEM → terrain/hydrology features
│   ├── Flood_label_extraction_GEE.js    # Sentinel-1 SAR flood labels (Earth Engine)
│   ├── Data_preprocessing.py            # legacy feature-engineering script (superseded for training)
│   ├── flood_pipeline.py                # leakage-safe imblearn Pipeline + custom transformers
│   ├── train.py                         # validation + MLflow tracking + model registry
│   └── Final_model.py                   # inference + Folium map / GeoJSON export
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── data-extraction.md
│   ├── data-preprocessing.md
│   ├── gee-flood-labels.md
│   ├── model-training.md
│   ├── mlflow-and-validation.md         # methodology + honest cross-validated results
│   └── Complete_Technical_Report.md
│
├── data/
│   ├── raw/          # unprocessed_dataset.csv, FloodLabels.csv, DEM.tif, rainfall_data.csv, boundaries
│   └── processed/    # engineered datasets
│
├── models/           # detector_xgb_binary.pkl, severity_xgb_binary.pkl, detector_threshold.txt
└── outputs/          # flood_risk_map.html, predictions (CSV/GeoJSON), plots/
```

---

## 🚀 Quick Start

```bash
# Environment
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train + validate (with MLflow tracking)

```bash
python src/train.py --config config.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db    # → http://127.0.0.1:5000
```

This runs both stages under repeated + spatial cross-validation, selects the recall-targeted threshold on out-of-fold probabilities, calibrates, evaluates the cascade, and registers `flood-detector` and `flood-severity` in the MLflow Model Registry.

### Full pipeline (from raw data)

```bash
# 1) Extract terrain/hydrology features from the DEM
python src/QGIS_DataExtraction_pipeline.py
# 2) Generate SAR flood labels in the Earth Engine Code Editor
#    (paste src/Flood_label_extraction_GEE.js at https://code.earthengine.google.com/)
# 3) Train + validate + track
python src/train.py --config config.yaml
# 4) Inference + interactive map
python src/Final_model.py            # → outputs/flood_risk_map.html
```

---

## 🔬 Validation & Experiment Tracking

The core engineering contribution. Details in [`docs/mlflow-and-validation.md`](docs/mlflow-and-validation.md).

- **Leakage-safe pipelines** — median imputation, rainfall PCA (9 yearly bands → 1 component), scaling and SMOTE are fit only on each fold's training rows, inside an `imblearn.Pipeline`. Stateless feature engineering (log/√ transforms, interaction and composite risk indices) uses a no-op `fit`.
- **Spatial cross-validation** — grid cells are binned into ~500 m UTM blocks and held out together via `GroupKFold`, so the model is never tested on a cell adjacent to one it trained on. Random-vs-spatial gap ≈ **0.14 ROC-AUC**.
- **Out-of-fold threshold selection** — the 90%-recall operating threshold is chosen on out-of-fold probabilities, not a slice of the training set.
- **Probability calibration** — isotonic calibration makes predicted probabilities (and the threshold) meaningful; tracked via Brier score.
- **Cascade evaluation** — severity is scored through the detector's decisions, so missed floods count as errors (real operational metric, not per-stage-in-isolation).
- **MLflow** — one parent run + nested `detector` / `severity` / `cascade` runs; params, metrics, plots and versioned models logged to a SQLite backend.

---

## 🔎 Technical Highlights

- **Hierarchical modeling** decouples "is there a flood?" (recall-critical) from "how severe?" (precision-relevant), letting each stage be tuned to its own cost profile.
- **Feature engineering** — 16 model features from 20+ raw variables: flow accumulation, Topographic Wetness Index, distance-to-river, rainfall PCA, and domain composite indices.
- **Imbalance handling** — SMOTE (in-fold) for the severity stage; recall-oriented thresholding for the detector.
- **Design insight** — XGBoost is invariant to monotonic feature transforms, so the leakage that materially mattered here was **PCA, imputation, SMOTE and threshold selection**, not scaling.

---

## 🗺️ Roadmap

- [ ] Nested Optuna hyperparameter search inside the inner CV loop (logged to MLflow)
- [ ] Strengthen the severity stage (near-chance spatially) with more events/years and spatial features
- [ ] SHAP-based explainability per prediction
- [ ] Inference API + automated retraining trigger on drift

---

## 📚 References

- [Sentinel-1 SAR Handbook](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1)
- [Google Earth Engine Guides](https://developers.google.com/earth-engine/guides)
- [XGBoost Documentation](https://xgboost.readthedocs.io/) · [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [SAR Flood Detection Review](https://doi.org/10.3390/rs13061117)

---

## 📄 License

MIT License — see `LICENSE`.

**Author:** Mohit Chauhan — Data Scientist / ML Engineer, IIT Roorkee
