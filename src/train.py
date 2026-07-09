"""
train.py
========
Validation-first training entry point with MLflow tracking.

What it does, and why
---------------------
1. Loads RAW features + labels and builds targets (no pre-transformed CSV).
2. Evaluates the flood **detector** two ways:
     * RepeatedStratifiedKFold  -> stable mean +/- std (random CV)
     * Spatial GroupKFold       -> honest generalisation across the map
   The gap between them quantifies spatial-autocorrelation optimism.
3. Selects the recall-targeted decision threshold on out-of-fold probabilities
   (not on a slice of the training data), optionally after isotonic calibration.
4. Evaluates the **severity** classifier the same way (SMOTE fit in-fold).
5. Evaluates the **cascade** end-to-end (missed floods count as errors).
6. Fits the final production pipelines on all data and logs params, metrics,
   plots and the models (with Model Registry) to MLflow.

Run:  python src/train.py            (uses ./config.yaml)
View: mlflow ui --backend-store-uri ./mlruns
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make flood_pipeline importable

import numpy as np
import pandas as pd
import yaml

import mlflow
import mlflow.sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    GroupKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from flood_pipeline import (
    feature_frame,
    load_raw,
    make_detector_pipeline,
    make_severity_pipeline,
    make_targets,
    spatial_groups,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def data_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:12]


def select_threshold(y_true, y_prob, recall_target: float) -> float:
    """Lowest-precision-cost threshold that still reaches the recall target;
    fall back to best-F1 if the target is unreachable."""
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    thr = np.append(thr, 1.0)                       # align lengths
    ok = recall >= recall_target
    if ok.any():
        idx = np.argmax(np.where(ok, precision, -1))    # highest precision among ok
        return float(thr[idx])
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thr[int(np.nanargmax(f1))])


def metrics_at(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def cv_summary(pipe, X, y, cv, groups=None) -> dict:
    """mean/std of threshold-free scores across folds."""
    scoring = ["roc_auc", "average_precision", "f1", "recall", "precision"]
    res = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=scoring, n_jobs=-1)
    out = {}
    for s in scoring:
        out[f"{s}_mean"] = float(np.mean(res[f"test_{s}"]))
        out[f"{s}_std"] = float(np.std(res[f"test_{s}"]))
    return out


def save_threshold_plots(y, oof_prob, threshold, out_prefix) -> list[str]:
    """Honest, cross-validated PR / ROC / threshold-sweep curves from OOF probs."""
    if not HAS_PLT:
        return []
    from sklearn.metrics import roc_curve

    paths = []
    precision, recall, _ = precision_recall_curve(y, oof_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR curve (out-of-fold)"); plt.grid(True)
    p = f"{out_prefix}_pr_curve.png"; plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(); paths.append(p)

    fpr, tpr, _ = roc_curve(y, oof_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, oof_prob):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (out-of-fold)")
    plt.legend(); plt.grid(True)
    p = f"{out_prefix}_roc_curve.png"; plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(); paths.append(p)

    ts = np.linspace(0.01, 0.99, 99)
    prec = [precision_score(y, (oof_prob >= t).astype(int), zero_division=0) for t in ts]
    rec = [recall_score(y, (oof_prob >= t).astype(int), zero_division=0) for t in ts]
    f1s = [f1_score(y, (oof_prob >= t).astype(int), zero_division=0) for t in ts]
    plt.figure(figsize=(7, 5))
    plt.plot(ts, prec, label="Precision"); plt.plot(ts, rec, label="Recall"); plt.plot(ts, f1s, label="F1")
    plt.axvline(threshold, color="k", ls="--", label=f"threshold={threshold:.3f}")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Threshold sweep (out-of-fold)")
    plt.legend(); plt.grid(True)
    p = f"{out_prefix}_threshold_sweep.png"; plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(); paths.append(p)
    return paths


def log_model_and_register(pipe, artifact_path, registered_name, input_example):
    """Log an sklearn/imblearn pipeline; register in the Model Registry if the
    backend supports it (never let a registry/serialization quirk crash the run).

    cloudpickle (not MLflow 3.x's default skops) is required because the pipeline
    contains custom transformers and an XGBoost Booster. float64 input example
    avoids int-column schema errors when inference data has NaNs."""
    info = mlflow.sklearn.log_model(
        pipe,
        name=artifact_path,
        input_example=input_example.astype("float64"),
        serialization_format="cloudpickle",
    )
    try:
        mlflow.register_model(info.model_uri, registered_name)
    except Exception as exc:   # noqa: BLE001
        print(f"[registry] skipped ({exc.__class__.__name__}: {exc}); model still logged.")
    return info


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(config_path: str):
    cfg = load_config(config_path)
    v = cfg["validation"]
    rs = v["random_state"]
    os.makedirs(cfg["paths"]["outputs_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    plots_dir = os.path.join(cfg["paths"]["outputs_dir"], "plots")
    os.makedirs(plots_dir, exist_ok=True)

    mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    # ---- data ----
    df = load_raw(cfg)
    X = feature_frame(df, cfg)
    y_bin, y_sev_full = make_targets(df, cfg)
    groups = spatial_groups(df, v["spatial_block_m"])

    rep_cv = RepeatedStratifiedKFold(n_splits=v["cv_folds"], n_repeats=v["cv_repeats"], random_state=rs)
    skf = StratifiedKFold(n_splits=v["cv_folds"], shuffle=True, random_state=rs)
    gkf = GroupKFold(n_splits=min(v["cv_folds"], pd.Series(groups).nunique()))

    with mlflow.start_run(run_name="hierarchical-pipeline") as parent:
        mlflow.set_tags({"project": "flood-prediction", "stage": "validation+training"})
        mlflow.log_params({
            "data_hash": data_hash(df),
            "n_samples": len(df),
            "n_features_raw": X.shape[1],
            "class_balance_binary": json.dumps(y_bin.value_counts().to_dict()),
            "class_balance_multiclass": json.dumps(df[cfg["labels"]["multiclass"]].value_counts().to_dict()),
            "cv_folds": v["cv_folds"],
            "cv_repeats": v["cv_repeats"],
            "spatial_block_m": v["spatial_block_m"],
            "n_spatial_groups": pd.Series(groups).nunique(),
            "recall_target": v["recall_target"],
            "calibrate": v["calibrate"],
        })

        # ================= DETECTOR =================
        with mlflow.start_run(run_name="detector", nested=True):
            mlflow.log_params({f"det_{k}": val for k, val in cfg["detector_params"].items()})

            rand = cv_summary(make_detector_pipeline(cfg), X, y_bin, rep_cv)
            spat = cv_summary(make_detector_pipeline(cfg), X, y_bin, gkf, groups=groups)
            mlflow.log_metrics({f"randcv_{k}": val for k, val in rand.items()})
            mlflow.log_metrics({f"spatialcv_{k}": val for k, val in spat.items()})
            # optimism from spatial autocorrelation
            mlflow.log_metric("optimism_recall_gap", rand["recall_mean"] - spat["recall_mean"])
            mlflow.log_metric("optimism_auc_gap", rand["roc_auc_mean"] - spat["roc_auc_mean"])

            # out-of-fold probabilities for threshold + calibration
            oof = cross_val_predict(make_detector_pipeline(cfg), X, y_bin, cv=skf,
                                    method="predict_proba", n_jobs=-1)[:, 1]
            prod_prob = oof
            if v["calibrate"]:
                cal = CalibratedClassifierCV(make_detector_pipeline(cfg), method="isotonic", cv=skf)
                oof_cal = cross_val_predict(cal, X, y_bin, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
                mlflow.log_metric("brier_uncalibrated", brier_score_loss(y_bin, oof))
                mlflow.log_metric("brier_calibrated", brier_score_loss(y_bin, oof_cal))
                prod_prob = oof_cal

            threshold = select_threshold(y_bin, prod_prob, v["recall_target"])
            at = metrics_at(y_bin, prod_prob, threshold)
            mlflow.log_metric("chosen_threshold", threshold)
            mlflow.log_metrics({f"oof_{k}": val for k, val in at.items()})
            cm = confusion_matrix(y_bin, (prod_prob >= threshold).astype(int))
            mlflow.log_dict({"confusion_matrix": cm.tolist()}, "detector_confusion.json")

            for p in save_threshold_plots(y_bin, prod_prob, threshold, os.path.join(plots_dir, "detector")):
                mlflow.log_artifact(p, "plots")

            # persist threshold with the model
            thr_path = os.path.join(cfg["paths"]["models_dir"], "detector_threshold.txt")
            with open(thr_path, "w") as fh:
                fh.write(str(threshold))
            mlflow.log_artifact(thr_path)

            # final production detector (calibrated) fit on ALL data
            final_det = (CalibratedClassifierCV(make_detector_pipeline(cfg), method="isotonic", cv=skf)
                         if v["calibrate"] else make_detector_pipeline(cfg))
            final_det.fit(X, y_bin)
            log_model_and_register(final_det, "detector",
                                   cfg["mlflow"]["detector_registered_name"], X.head(3))

            print(f"[detector] random recall={rand['recall_mean']:.3f}  "
                  f"spatial recall={spat['recall_mean']:.3f}  "
                  f"gap={rand['recall_mean']-spat['recall_mean']:+.3f}  thr={threshold:.3f}")

        # ================= SEVERITY (floods only) =================
        flood_mask = y_bin.to_numpy() == 1
        X_sev = X.loc[flood_mask].reset_index(drop=True)
        y_sev = y_sev_full.loc[flood_mask].reset_index(drop=True)
        groups_sev = groups[flood_mask]

        with mlflow.start_run(run_name="severity", nested=True):
            mlflow.log_params({f"sev_{k}": val for k, val in cfg["severity_params"].items()})
            gkf_sev = GroupKFold(n_splits=min(v["cv_folds"], pd.Series(groups_sev).nunique()))

            rand_s = cv_summary(make_severity_pipeline(cfg), X_sev, y_sev, rep_cv)
            spat_s = cv_summary(make_severity_pipeline(cfg), X_sev, y_sev, gkf_sev, groups=groups_sev)
            mlflow.log_metrics({f"randcv_{k}": val for k, val in rand_s.items()})
            mlflow.log_metrics({f"spatialcv_{k}": val for k, val in spat_s.items()})

            sev_oof = cross_val_predict(make_severity_pipeline(cfg), X_sev, y_sev, cv=skf,
                                        method="predict_proba", n_jobs=-1)[:, 1]
            mlflow.log_metrics({f"oof_{k}": val for k, val in metrics_at(y_sev, sev_oof, 0.5).items()})

            final_sev = make_severity_pipeline(cfg).fit(X_sev, y_sev)
            log_model_and_register(final_sev, "severity",
                                   cfg["mlflow"]["severity_registered_name"], X_sev.head(3))
            print(f"[severity] random f1={rand_s['f1_mean']:.3f}  spatial f1={spat_s['f1_mean']:.3f}")

        # ================= CASCADE (end-to-end, floods) =================
        with mlflow.start_run(run_name="cascade", nested=True):
            det_pred = (prod_prob >= threshold).astype(int)          # OOF detector decision
            # 3-class cascade prediction over TRUE floods: missed flood -> class 0 (none)
            fc = df[cfg["labels"]["multiclass"]].to_numpy()
            sev_pred_full = np.zeros(len(df), dtype=int)
            sev_pred_full[flood_mask] = np.where(sev_oof >= 0.5, 2, 1)
            cascade_pred = np.where(det_pred == 1, sev_pred_full, 0)   # undetected -> none

            true_flood = fc > 0
            missed = ((det_pred == 0) & true_flood).sum() / max(true_flood.sum(), 1)
            cascade_acc = accuracy_score(fc[true_flood], cascade_pred[true_flood])
            cascade_f1 = f1_score(fc[true_flood], cascade_pred[true_flood], average="macro")
            mlflow.log_metrics({
                "missed_flood_rate": float(missed),
                "cascade_accuracy_on_floods": float(cascade_acc),
                "cascade_macro_f1_on_floods": float(cascade_f1),
            })
            print(f"[cascade] missed floods={missed:.3f}  3-class acc(on floods)={cascade_acc:.3f}")

        print(f"\nParent run: {parent.info.run_id}")
        print("View with:  mlflow ui --backend-store-uri", cfg["paths"]["mlflow_uri"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
