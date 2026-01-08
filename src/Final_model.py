"""
hierarchical_flood_pipeline_visual.py

- Hierarchical flood pipeline: binary detector (Flood? 0/1) -> severity (Low/High)
- Adds visualizations: PR curve, ROC curve, threshold plots, accuracy vs threshold
- Exports interactive Folium map (HTML) of predictions for the site
- Saves models and tuned threshold for reproducibility

Required packages:
pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib matplotlib seaborn folium geopandas

Adjust file paths and column names if needed.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import geopandas as gpd
from shapely.geometry import Point
from folium.plugins import HeatMap

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, roc_auc_score, precision_recall_fscore_support
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# --------------------------
# Config / paths
# --------------------------
RANDOM_STATE = 42
INPUT_CSV = "ML_ready_dataset.csv"          # your processed dataset
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Map outputs
MAP_HTML = os.path.join(OUT_DIR, "flood_risk_map.html")
PLOT_PREFIX = os.path.join(OUT_DIR, "detector_plots")
DETECTOR_MODEL_PATH = os.path.join(OUT_DIR, "detector_xgb_binary.pkl")
SEVERITY_MODEL_PATH = os.path.join(OUT_DIR, "severity_xgb_binary.pkl")
THRESH_PATH = os.path.join(OUT_DIR, "detector_threshold.txt")
PRED_CSV = os.path.join(OUT_DIR, "hierarchical_flood_predictions.csv")

# --------------------------
# 1. Load & basic checks
# --------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded dataset:", INPUT_CSV, "rows:", len(df))
# Ensure required cols exist
required = {"floodClass", "latitude", "longitude"}
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# --------------------------
# 2. Hierarchical labels
# --------------------------
# binary flood label: any flood (1 if Low or High)
df["flood_any"] = df["floodClass"].apply(lambda x: 1 if int(x) in [1, 2] else 0)

# severity dataset (only true floods). severity_bin: 0 = Low, 1 = High
df_severity = df[df["flood_any"] == 1].copy()
df_severity["severity_bin"] = df_severity["floodClass"].apply(lambda x: 0 if int(x) == 1 else 1)

print("Class counts (multi):", df["floodClass"].value_counts().to_dict())
print("Class counts (binary flood_any):", df["flood_any"].value_counts().to_dict())
print("Severity counts (Low=0,High=1):", df_severity["severity_bin"].value_counts().to_dict())

# --------------------------
# 3. Feature list (auto) - drop labels only
# --------------------------
LABEL_MULTI = "floodClass"
LABEL_BIN = "flood_any"
SEV_LABEL = "severity_bin"

# Use all columns except the labels and geographic columns remain as features if desired
# If some columns should be excluded (IDs, text), remove them here
exclude = {LABEL_MULTI, LABEL_BIN, SEV_LABEL}
# Keep lat/lon only for mapping; they can also be used as features if you want spatial patterns
features = [c for c in df.columns if c not in exclude]
# Optionally remove lat/lon from features if you don't want spatial coords as model inputs:
# features = [c for c in features if c not in ["latitude", "longitude"]]

print("Number of features used:", len(features))

# --------------------------
# 4. Stage A: Detector (binary) training
# --------------------------
X = df[features]
y = df[LABEL_BIN].astype(int)

# stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Detector train/test sizes:", X_train.shape, X_test.shape)

# sample weights to handle imbalance
sw = compute_sample_weight(class_weight="balanced", y=y_train)

detector = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    tree_method="hist"
)
detector.fit(X_train, y_train, sample_weight=sw)

# default-threshold evaluation
y_prob_test = detector.predict_proba(X_test)[:, 1]
y_pred_test_default = (y_prob_test >= 0.5).astype(int)

print("\nDetector - Default threshold 0.5")
print("Accuracy:", accuracy_score(y_test, y_pred_test_default))
print(classification_report(y_test, y_pred_test_default, target_names=["No Flood","Flood"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_default))

# --------------------------
# 5. Threshold tuning using precision-recall on a validation fold
# --------------------------
# Hold out a small validation subset from training to tune threshold
X_train_sub, X_val, y_train_sub, y_val_sub = train_test_split(
    X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
)
# refit on subtrain for tuning stability
sw_sub = compute_sample_weight(class_weight="balanced", y=y_train_sub)
detector.fit(X_train_sub, y_train_sub, sample_weight=sw_sub)

y_val_prob = detector.predict_proba(X_val)[:, 1]
precision, recall, pr_thresholds = precision_recall_curve(y_val_sub, y_val_prob)

# choose threshold by recall target, fallback to best F1
RECALL_TARGET = 0.90
candidates = [(t, p, r) for p, r, t in zip(precision, recall, np.append(pr_thresholds, 1.0))]
good = [(t, p, r) for (t, p, r) in candidates if r >= RECALL_TARGET]
if good:
    chosen = max(good, key=lambda x: x[1])   # highest precision among good
    chosen_threshold = chosen[0]
    print(f"\nChosen threshold to reach recall >= {RECALL_TARGET}: {chosen_threshold:.3f}, precision {chosen[1]:.3f}, recall {chosen[2]:.3f}")
else:
    # choose best F1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = np.nanargmax(f1_scores)
    chosen_threshold = np.append(pr_thresholds, 1.0)[best_idx]
    print(f"\nNo threshold reaches recall {RECALL_TARGET}. Choosing threshold with best F1: {chosen_threshold:.3f}")

# Evaluate tuned threshold on test set
y_pred_test_tuned = (y_prob_test >= chosen_threshold).astype(int)
print("\nDetector - Tuned threshold results")
print("Threshold:", chosen_threshold)
print("Accuracy:", accuracy_score(y_test, y_pred_test_tuned))
print(classification_report(y_test, y_pred_test_tuned, target_names=["No Flood", "Flood"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_tuned))

# save detector and threshold
joblib.dump(detector, DETECTOR_MODEL_PATH)
with open(THRESH_PATH, "w") as fh:
    fh.write(str(chosen_threshold))

# --------------------------
# 6. Detector visualizations:
#    - Precision-Recall curve
#    - ROC curve
#    - Threshold vs precision/recall/accuracy plots
# --------------------------
# PR curve
plt.figure(figsize=(7,6))
plt.plot(recall, precision, label=f'PR curve (val)')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (validation)")
plt.grid(True)
plt.legend()
plt.savefig(PLOT_PREFIX + "_pr_curve.png", dpi=150, bbox_inches='tight')

# ROC curve on validation
fpr, tpr, roc_thr = roc_curve(y_val_sub, y_val_prob)
roc_auc = roc_auc_score(y_val_sub, y_val_prob)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (validation)")
plt.grid(True)
plt.legend()
plt.savefig(PLOT_PREFIX + "_roc_curve.png", dpi=150, bbox_inches='tight')

# Threshold sweep: compute precision/recall/accuracy across many thresholds (from val set)
thr_vals = np.linspace(0.0, 1.0, 101)
prec_vals, rec_vals, acc_vals, f1_vals = [], [], [], []
for t in thr_vals:
    pred = (y_val_prob >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_val_sub, pred, average='binary', zero_division=0)
    acc = (pred == y_val_sub).mean()
    prec_vals.append(p); rec_vals.append(r); acc_vals.append(acc); f1_vals.append(f1)

plt.figure(figsize=(8,6))
plt.plot(thr_vals, prec_vals, label='Precision')
plt.plot(thr_vals, rec_vals, label='Recall')
plt.plot(thr_vals, acc_vals, label='Accuracy')
plt.plot(thr_vals, f1_vals, label='F1')
plt.axvline(chosen_threshold, color='k', linestyle='--', label=f'Chosen threshold {chosen_threshold:.3f}')
plt.xlabel("Threshold")
plt.ylabel("Metric")
plt.title("Threshold sweep (validation) - Precision / Recall / Accuracy / F1")
plt.legend();
plt.grid(True)
plt.savefig(PLOT_PREFIX + "_threshold_sweep.png", dpi=150, bbox_inches='tight')

# --------------------------
# 7. Stage B: Severity classifier (Low(0) vs High(1)) training & evaluation
# --------------------------
# Use df_severity features
X_sev = df_severity[features]
y_sev = df_severity[SEV_LABEL].astype(int)

# split for severity
X_sev_train, X_sev_test, y_sev_train, y_sev_test = train_test_split(
    X_sev, y_sev, test_size=0.20, random_state=RANDOM_STATE, stratify=y_sev
)
print("\nSeverity train/test sizes:", X_sev_train.shape, X_sev_test.shape)

# If small / imbalanced, apply SMOTE
if len(X_sev_train) > 20:
    sm = SMOTE(k_neighbors=3, random_state=RANDOM_STATE)
    X_sev_train_bal, y_sev_train_bal = sm.fit_resample(X_sev_train, y_sev_train)
    print("Severity after SMOTE:", np.bincount(y_sev_train_bal))
else:
    X_sev_train_bal, y_sev_train_bal = X_sev_train, y_sev_train

# severity class weights & sample weights
if len(y_sev_train) > 0:
    sev_cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_sev_train), y=y_sev_train)
    sev_map = {cls: w for cls, w in zip(np.unique(y_sev_train), sev_cw)}
else:
    sev_map = {0:1.0, 1:1.0}

sample_weights_sev = np.array([sev_map[int(c)] for c in y_sev_train_bal])

severity_clf = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    tree_method="hist"
)
if len(X_sev_train_bal) > 0:
    severity_clf.fit(X_sev_train_bal, y_sev_train_bal, sample_weight=sample_weights_sev)
    joblib.dump(severity_clf, SEVERITY_MODEL_PATH)

# Evaluate severity on true-flood test subset
if len(X_sev_test) > 0:
    y_sev_pred = severity_clf.predict(X_sev_test)
    print("\nSeverity classification report (Low=0, High=1):")
    print(classification_report(y_sev_test, y_sev_pred, target_names=["Low","High"]))
    print("Confusion matrix:\n", confusion_matrix(y_sev_test, y_sev_pred))

# --------------------------
# 8. Deploy: predict on full dataset using detector threshold and severity model
# --------------------------
X_all = df[features]
probs_all = detector.predict_proba(X_all)[:, 1]
pred_any_all = (probs_all >= chosen_threshold).astype(int)

df_out = df.copy()
df_out["detector_prob"] = probs_all
df_out["detector_pred"] = pred_any_all

# apply severity model to points flagged as flood
sev_preds = np.full(len(df_out), -1, dtype=int)  # -1 => not flagged
mask = df_out["detector_pred"] == 1
if mask.sum() > 0 and 'severity_clf' in globals():
    # use model to predict severity only for flagged points
    sev_preds_mask = severity_clf.predict(df_out.loc[mask, features])
    sev_preds[mask.values] = sev_preds_mask
df_out["severity_pred"] = sev_preds

# Save CSV for GIS or analysis
df_out.to_csv(PRED_CSV, index=False)
print("\nSaved predictions to:", PRED_CSV)


# 9. Build interactive map (Folium)
#    - Points colored by detector_prob (heat-like) and shaped by severity_pred
#    - Save as HTML
# --------------------------

df_map= pd.read_csv("final_dataset_processed.csv")

#The lat/lon columns currently contain UTM centroids (in meters)
print("\n=== DEBUGGING COORDINATE VALUES ===")
print(f"First 3 rows of coordinate columns:")
print(df_map[["latitude", "longitude"]].head(3))
print(f"\nData types:")
print(df_map[["latitude", "longitude"]].dtypes)
print(f"\nMin/Max values:")
print(f"Latitude - Min: {df_map['latitude'].min()}, Max: {df_map['latitude'].max()}")
print(f"Longitude - Min: {df_map['longitude'].min()}, Max: {df_map['longitude'].max()}")

# Store unscaled coordinates
df_out["lat_wgs84"] = df_map["latitude"]
df_out["lon_wgs84"] = df_map["longitude"]

# Calculate map center
mean_lat = float(df_out["lat_wgs84"].mean())
mean_lon = float(df_out["lon_wgs84"].mean())

# Professional hydrology color scales
def prob_color(p):
    # blue gradient from low → high probability
    return (
        "rgba(198,219,239,0.6)" if p < 0.2 else
        "rgba(158,202,225,0.7)" if p < 0.4 else
        "rgba(107,174,214,0.8)" if p < 0.6 else
        "rgba(49,130,189,0.9)"  if p < 0.8 else
        "rgba(8,81,156,1.0)"
    )

def severity_color(s):
    # -1 = Not flood, 0 = Low, 1 = High
    if s == -1: return "gray"
    if s == 0:  return "#ffcc00"   # Low = yellow
    if s == 1:  return "#ff3300"   # High = red

# Create Folium map
m = folium.Map(
    location=[mean_lat, mean_lon],
    zoom_start=12,
    tiles="CartoDB positron",
    control_scale=True
)

folium.TileLayer("Stamen Terrain").add_to(m)
folium.TileLayer("OpenStreetMap").add_to(m)

# Heatmap of detector probability (continuous)
heatmap_data = df_out.apply(
    lambda r: [r["lat_wgs84"], r["lon_wgs84"], r["detector_prob"]],
    axis=1
).tolist()

HeatMap(
    heatmap_data,
    radius=15,
    blur=25,
    max_zoom=14,
    min_opacity=0.5,
    name="Flood Probability Heatmap"
).add_to(m)

#  Add Severity Layer (Colored Circle Markers)
severity_layer = folium.FeatureGroup(name="Severity Classification (Low/High)")

for _, row in df_out.iterrows():
    lat = row["lat_wgs84"]
    lon = row["lon_wgs84"]
    sev = row["severity_pred"]
    p = row["detector_prob"]

    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=severity_color(sev),
        fill=True,
        fill_color=severity_color(sev),
        fill_opacity=0.9,
        popup=folium.Popup(
            f"<b>Flood Probability:</b> {p:.3f}<br>"
            f"<b>Severity:</b> {'None' if sev==-1 else ('Low' if sev==0 else 'High')}",
            max_width=250
        ),
        tooltip=f"Severity: {'None' if sev==-1 else ('Low' if sev==0 else 'High')} | Prob={p:.2f}"
    ).add_to(severity_layer)

severity_layer.add_to(m)

# Color function for severity
def sev_color(v):
    if v == -1:
        return "#AAAAAA"   # not flagged = gray
    if v == 0:
        return "#FFA500"   # Low = orange
    if v == 1:
        return "#FF0000"   # High = red
    return "#000000"

# Add circle markers for each prediction
print(f"Adding {len(df_out)} points to map...")
for i, row in df_out.iterrows():
    lat = row["lat_wgs84"]
    lon = row["lon_wgs84"]
    prob = float(row["detector_prob"])
    sev = int(row["severity_pred"]) if pd.notna(row["severity_pred"]) else -1
    color = sev_color(sev)
    
    popup = folium.Popup(
        html=f"<b>Point ID:</b> {i}<br>"
             f"<b>UTM X:</b> {row['longitude']:.2f}<br>"
             f"<b>UTM Y:</b> {row['latitude']:.2f}<br>"
             f"<b>WGS84:</b> {lat:.6f}, {lon:.6f}<br>"
             f"<b>Flood Prob:</b> {prob:.3f}<br>"
             f"<b>Flood Pred:</b> {'Yes' if row['detector_pred'] == 1 else 'No'}<br>"
             f"<b>Severity:</b> {'High' if sev == 1 else 'Low' if sev == 0 else 'N/A'}",
        max_width=320
    )
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=min(0.9, 0.3 + prob * 0.7),
        weight=1,
        popup=popup
    ).add_to(m)

# Add legend
legend_html = """
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: auto; 
     border: 2px solid grey; z-index: 9999; font-size: 14px;
     background-color: white; opacity: 0.95; padding: 10px;
     border-radius: 5px;">
<b>Flood Risk Legend</b><br><br>
<span style="color:#FF0000; font-size:18px;">●</span>&nbsp;High Severity<br>
<span style="color:#FFA500; font-size:18px;">●</span>&nbsp;Low Severity<br>
<span style="color:#AAAAAA; font-size:18px;">●</span>&nbsp;No Flood Risk<br><br>
<i style="font-size:11px;">Opacity indicates probability</i>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Save map
folium.LayerControl().add_to(m)

m.save(MAP_HTML)
print("\nHydrology dashboard exported to:", MAP_HTML)

# Update predictions CSV with WGS84 coordinates
df_out.to_csv(PRED_CSV, index=False)
print(f"✓ Updated predictions CSV with WGS84 coordinates: {PRED_CSV}")
print(f"\n=== After predictions (df_out) ===")
print(f"Latitude - Min: {df_out['latitude'].min()}, Max: {df_out['latitude'].max()}")
print(f"Longitude - Min: {df_out['longitude'].min()}, Max: {df_out['longitude'].max()}")
print(f"Total rows: {len(df_out)}")

# Save GeoJSON with correct CRS
try:
    geoms = [Point(xy) for xy in zip(df_out["lon_wgs84"], df_out["lat_wgs84"])]
    gdf = gpd.GeoDataFrame(df_out, geometry=geoms, crs="EPSG:4326")
    geojson_path = os.path.join(OUT_DIR, "hierarchical_flood_predictions.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"✓ Saved GeoJSON to: {geojson_path}")
except Exception as e:
    print(f"⚠ Could not save GeoJSON: {e}")

print(f"\n{'='*60}")
print(f"All outputs saved to: {OUT_DIR}")
print(f"Open {MAP_HTML} in your browser to view the flood risk map!")
print(f"{'='*60}")


# Also export GeoJSON for GIS use
try:
    gdf = gpd.GeoDataFrame(df_out, geometry=gpd.points_from_xy(df_out.longitude, df_out.latitude), crs="EPSG:4326")
    geojson_path = os.path.join(OUT_DIR, "hierarchical_flood_predictions.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    print("Saved GeoJSON to:", geojson_path)
except Exception as e:
    print("Could not save GeoJSON:", e)

print("\nAll done. Plots and map written to", OUT_DIR)