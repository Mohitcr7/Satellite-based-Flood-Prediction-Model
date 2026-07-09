"""
flood_pipeline.py
=================
Leakage-safe data loading + preprocessing for the hierarchical flood model.

Why this file exists
--------------------
The original `Data_preprocessing.py` fit the imputer, PCA and StandardScaler on
the *entire* dataset and wrote a CSV that `Final_model.py` then split into
train/test.  That leaks test-set statistics into training and inflates every
reported metric.

Here, ALL stateful preprocessing (median imputation, rainfall PCA, scaling,
SMOTE) lives inside a single scikit-learn / imbalanced-learn ``Pipeline`` so it
is re-fit from scratch on the training rows of every CV fold.  Stateless,
row-wise feature engineering (logs, ratios, composite indices) carries no
leakage and is done in a no-op-``fit`` transformer.

Public API
----------
load_raw(cfg)                     -> merged raw DataFrame (features + floodClass)
make_targets(df, cfg)             -> (y_binary, y_severity_full)
feature_frame(df, cfg)            -> model-input columns only (raw, untransformed)
spatial_groups(df, block_m)       -> group id per row for spatial GroupKFold
make_detector_pipeline(cfg, ...)  -> imblearn Pipeline ending in XGBClassifier
make_severity_pipeline(cfg)       -> imblearn Pipeline with in-fold SMOTE
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Keep DataFrames flowing through the pipeline so the named-column feature
# engineering below can reference columns by name (sklearn >= 1.2).
set_config(transform_output="pandas")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_raw(cfg: dict) -> pd.DataFrame:
    """Merge untransformed features with SAR flood labels on the grid bbox."""
    feats = pd.read_csv(cfg["paths"]["raw_features"])
    labels = pd.read_csv(cfg["paths"]["labels"])
    df = pd.merge(feats, labels, how="inner", on=cfg["merge_keys"])
    if df.empty:
        raise ValueError("Feature/label merge produced 0 rows — check merge_keys.")
    return df


def make_targets(df: pd.DataFrame, cfg: dict):
    """Binary detector target and (floods-only) severity target."""
    fc = df[cfg["labels"]["multiclass"]].astype(int)
    y_binary = fc.isin([1, 2]).astype(int)          # flood_any
    y_severity = (fc == 2).astype(int)              # 0=mild, 1=severe (valid where flood)
    return y_binary, y_severity


def feature_frame(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return only the raw, untransformed model-input columns."""
    drop = set(cfg["drop_columns"]) | {cfg["labels"]["multiclass"]}
    return df.drop(columns=[c for c in drop if c in df.columns], errors="ignore").copy()


def spatial_groups(df: pd.DataFrame, block_m: int) -> np.ndarray:
    """Group id from UTM grid position so whole spatial blocks are held out
    together in GroupKFold — defeats the neighbouring-cell autocorrelation that
    makes a random split look better than the model really is."""
    gx = (df["X_min"] // block_m).astype(int)
    gy = (df["Y_min"] // block_m).astype(int)
    return (gx.astype(str) + "_" + gy.astype(str)).to_numpy()


# --------------------------------------------------------------------------- #
# Custom transformers
# --------------------------------------------------------------------------- #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, row-wise feature engineering (no fitted parameters => no leakage).

    Mirrors the transforms in the original Data_preprocessing.py: log1p on
    skewed positive features, non-linear terms, domain interaction/composite
    indices, then drops collinear raw columns.  The Hillshade binning step is
    intentionally omitted (weak feature; trees split Hillshade directly).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        rain = self.cfg["features"]["rainfall_pca_columns"]

        # log1p on skewed, non-negative columns
        for c in rain + ["Distance_to_River"]:
            if c in X.columns:
                X[c] = np.log1p(X[c].clip(lower=0))

        # non-linear terms
        if "Distance_to_River" in X.columns:
            X["Distance_sqrt"] = np.sqrt(np.abs(X["Distance_to_River"]) + 1)
        X["Hillshade_pow2"] = X["Hillshade"] ** 2

        # domain interactions / composite indices (fixed weights -> no leakage)
        X["Rainfall_Elevation_Risk"] = X["rainfall_2024"] * X["Elevation"]
        X["Distance_Elevation_Risk"] = X["Distance_sqrt"] / (1 + np.abs(X["Elevation"]))
        X["flood_risk_index"] = (
            0.35 * X["rainfall_2024"]
            + 0.30 * X["Flow_accumulation"]
            + 0.20 * (1 / (1 + X["Elevation"]))
            + 0.15 * (1 / (1 + X["Distance_sqrt"]))
        )
        X["Terrain_Drainage_Efficiency"] = np.abs(X["Drainage_direction"] - X["Aspect"])
        X["Terrain_Interaction_3Way"] = X["Hillshade"] * X["Aspect"] * X["Elevation"]

        drop = self.cfg["features"]["drop_after_engineering"]
        return X.drop(columns=[c for c in drop if c in X.columns], errors="ignore")


class RainfallPCA(BaseEstimator, TransformerMixin):
    """PCA over the yearly-rainfall block, fit on TRAIN rows only (per fold).

    Appends Rainfall_PC1..PCn and drops the raw rainfall columns.  This is the
    step whose leakage actually matters for XGBoost — unlike scaling/log, a PCA
    rotation fit on the full dataset genuinely bleeds test information."""

    def __init__(self, columns, n_components: int = 1):
        self.columns = columns
        self.n_components = n_components

    def fit(self, X, y=None):
        self.cols_ = [c for c in self.columns if c in X.columns]
        self.pca_ = PCA(n_components=self.n_components, random_state=0).fit(X[self.cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        comps = np.asarray(self.pca_.transform(X[self.cols_]))  # global pandas-output -> force ndarray
        for i in range(self.n_components):
            X[f"Rainfall_PC{i + 1}"] = comps[:, i]
        return X.drop(columns=self.cols_)


# --------------------------------------------------------------------------- #
# Pipeline factories
# --------------------------------------------------------------------------- #
def _preprocessing_steps(cfg: dict):
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("engineer", FeatureEngineer(cfg)),
        ("rain_pca", RainfallPCA(
            cfg["features"]["rainfall_pca_columns"],
            cfg["features"]["rainfall_pca_n"],
        )),
        ("scale", StandardScaler()),   # inert for XGBoost, kept for model-agnosticism
    ]


def make_detector_pipeline(cfg: dict, use_smote: bool = False) -> ImbPipeline:
    """Binary flood detector. Classes are near-balanced (599 vs 615) so we rely
    on threshold tuning rather than resampling by default."""
    steps = _preprocessing_steps(cfg)
    rs = cfg["validation"]["random_state"]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=rs, k_neighbors=5)))
    steps.append(("clf", XGBClassifier(**cfg["detector_params"])))
    return ImbPipeline(steps)


def make_severity_pipeline(cfg: dict) -> ImbPipeline:
    """Severity classifier (mild vs severe). SMOTE lives inside the pipeline so
    synthetic samples are only ever generated from a fold's training rows."""
    rs = cfg["validation"]["random_state"]
    steps = _preprocessing_steps(cfg)
    steps.append(("smote", SMOTE(random_state=rs, k_neighbors=5)))
    steps.append(("clf", XGBClassifier(**cfg["severity_params"])))
    return ImbPipeline(steps)
