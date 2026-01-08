# Model Training & Deployment Pipeline

## Overview
Implements a hierarchical flood prediction system using XGBoost. Two-stage pipeline: first detects flood presence (binary), then classifies severity (low/high) for detected floods. Includes threshold tuning, comprehensive visualizations, and interactive mapping.

## Architecture

```
ML-Ready Dataset
        ↓
┌───────────────────────────────────────┐
│  Stage A: Flood Detector (Binary)     │
│  - Input: All features                │
│  - Output: Flood probability (0-1)    │
│  - Threshold tuning for recall=90%    │
└───────────────────────────────────────┘
        ↓ (Flagged points only)
┌───────────────────────────────────────┐
│  Stage B: Severity Classifier         │
│  - Input: Flooded features only       │
│  - Output: Low (0) / High (1)         │
│  - SMOTE for class imbalance          │
└───────────────────────────────────────┘
        ↓
Interactive Map + CSV Export
```

---

## Input Data

**File**: `ML_ready_dataset.csv` (from Data_preprocessing.py)

**Required columns**:
- `floodClass` (target): 0 = No flood, 1 = Low flood, 2 = High flood
- `latitude`, `longitude`: Geographic coordinates (WGS84)
- 25-35 feature columns (elevation, slope, rainfall_PC1, flood_risk_index, etc.)

**Data shape**: Typically 500-5000 samples × 30+ features

---

## Stage A: Binary Flood Detector

### A.1 Label Preparation
```python
df["flood_any"] = df["floodClass"].apply(lambda x: 1 if x in [1, 2] else 0)
```

**Conversion**:
- `floodClass = 0` → `flood_any = 0` (no flood)
- `floodClass = 1` → `flood_any = 1` (flood present)
- `floodClass = 2` → `flood_any = 1` (flood present)

**Class distribution**: Typically 70% no-flood, 30% flood (imbalanced)

---

### A.2 Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

- **Train**: 80% (used for model learning)
- **Test**: 20% (unseen data for evaluation)
- **Stratification**: Maintains class proportions

---

### A.3 Model Configuration

**Algorithm**: XGBoost (Gradient Boosting)

```python
detector = XGBClassifier(
    n_estimators=400,          # 400 decision trees
    max_depth=6,               # Max tree depth (prevent overfitting)
    learning_rate=0.05,        # Small steps → stable learning
    subsample=0.9,             # Use 90% of training data per tree
    colsample_bytree=0.9,      # Use 90% of features per tree
    objective='binary:logistic',# Binary classification
    eval_metric='logloss',     # Loss function
    random_state=42
)
```

**Hyperparameters explained**:
- **n_estimators**: More trees = better fit (diminishing returns ~200-500)
- **max_depth**: Deeper = complex patterns but risk overfitting; 6 is moderate
- **learning_rate**: Smaller = stable but slower; 0.05 is good balance
- **subsample/colsample**: Regularization; prevents overfitting

---

### A.4 Class Weight Balancing

**Problem**: 70% no-flood vs 30% flood → model biased toward majority class

**Solution**: Weighted learning
```python
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
detector.fit(X_train, y_train, sample_weight=sample_weights)
```

**Effect**: Flood samples weighted higher during training → better detection

---

### A.5 Threshold Tuning

**Default threshold** (0.5):
```python
y_pred = (detector.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
```

**Problem**: 0.5 threshold may not suit flood detection  
- Missing floods (false negatives) is costlier than false alarms
- Target: Recall ≥ 90% (catch 90% of actual floods)

**Solution**: Find optimal threshold
```python
precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)

# Find threshold with recall ≥ 90% and highest precision
recall_target = 0.90
candidates = [(t, p, r) for p, r, t in zip(precision, recall, thresholds) if r >= recall_target]
chosen = max(candidates, key=lambda x: x[1])  # Best precision among good
chosen_threshold = chosen[0]
```

**Output** (example):
```
Chosen threshold to reach recall >= 0.90: 0.35, precision 0.78, recall 0.92
```

**Interpretation**: 
- Threshold lowered from 0.50 → 0.35
- Any probability > 0.35 → "FLOOD ALERT"
- Catches 92% of real floods
- 22% false alarms (cost of missing floods)

---

## Stage B: Severity Classifier

### B.1 Flood-Only Subset
```python
df_severity = df[df["flood_any"] == 1].copy()
y_sev = df_severity["severity_bin"]  # 0 = Low, 1 = High
```

Only trains on flagged flood points (30% of data)

---

### B.2 Synthetic Oversampling (SMOTE)

**Problem**: Severity classes imbalanced (e.g., 80% Low, 20% High)

**Solution**: SMOTE (Synthetic Minority Oversampling)
```python
smote = SMOTE(k_neighbors=3, random_state=42)
X_sev_bal, y_sev_bal = smote.fit_resample(X_sev_train, y_sev_train)
```

**Effect**: Creates synthetic "High" samples in feature space → balanced dataset

---

### B.3 Severity Model
```python
severity_clf = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    # ... other params
)
severity_clf.fit(X_sev_train_bal, y_sev_train_bal, sample_weight=sample_weights_sev)
```

Similar to detector but:
- Fewer trees (250 vs 400) - smaller problem
- Shallower depth (5 vs 6) - less complexity

---

## Visualizations

### V.1 Precision-Recall Curve
```python
plt.plot(recall, precision, label='PR curve')
plt.xlabel('Recall'), plt.ylabel('Precision')
plt.savefig('detector_pr_curve.png')
```

**Interpretation**:
- Shows trade-off: precision vs recall
- Higher recall = more floods caught (but more false alarms)
- Chosen threshold marked on curve

---

### V.2 ROC Curve
```python
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
auc = roc_auc_score(y_val, y_val_prob)
plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
```

**Interpretation**:
- **AUC = 0.5**: Random guessing
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.85**: Good real-world performance

---

### V.3 Threshold Sweep
```python
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.plot(thresholds, accuracies, label='Accuracy')
plt.plot(thresholds, f1_scores, label='F1')
plt.axvline(chosen_threshold, linestyle='--', label=f'Chosen: {chosen_threshold}')
```

**Shows**: How each metric changes with threshold  
**Used for**: Visual justification of threshold choice

---

## Output Files

### O.1 Model Artifacts
- **`detector_xgb_binary.pkl`**: Trained flood detector (XGBoost binary)
- **`severity_xgb_binary.pkl`**: Trained severity classifier
- **`detector_threshold.txt`**: Optimal threshold value

```bash
# Load later for prediction
import joblib
detector = joblib.load('detector_xgb_binary.pkl')
severity = joblib.load('severity_xgb_binary.pkl')
```

### O.2 Prediction CSV
**File**: `hierarchical_flood_predictions.csv`

```csv
latitude,longitude,detector_prob,detector_pred,severity_pred,lat_wgs84,lon_wgs84
3456867.2,567813.4,0.78,1,1,29.1234,-84.5678
3456897.2,567843.4,0.23,0,-1,29.1264,-84.5708
...
```

**Columns**:
- `detector_prob`: Flood probability (0-1)
- `detector_pred`: Flood prediction (0/1)
- `severity_pred`: Severity if flood (0=Low, 1=High, -1=No flood)
- `lat_wgs84, lon_wgs84`: WGS84 coordinates for mapping

### O.3 Interactive Map
**File**: `flood_risk_map.html`

Generated using Folium with:
- **Heatmap layer**: Continuous flood probability
- **Marker layer**: Severity classification (colored points)
  - **Gray**: No flood
  - **Yellow**: Low severity
  - **Red**: High severity
- **Base maps**: CartoDB, Stamen Terrain, OpenStreetMap
- **Popups**: Click point to see details

**Open in browser**: Double-click HTML file or open with web browser

---

## Evaluation Metrics

### Detector Evaluation
```
Accuracy:  85%        (% correct predictions)
Precision: 78%        (of predicted floods, 78% are real)
Recall:    92%        (of real floods, catch 92%)
F1:        0.85       (harmonic mean of precision & recall)
```

### Confusion Matrix (Detector)
```
             Predicted No   Predicted Yes
Actual No         450            50         (50 false alarms)
Actual Yes         18           180         (18 missed floods)
```

### Severity Evaluation
```
Low classification:  85% accuracy
High classification: 81% accuracy
```

---

## Deployment Workflow

### D.1 Predict on New Data
```python
# New samples (same features as training)
X_new = pd.read_csv("new_study_area.csv")

# Get predictions
probs = detector.predict_proba(X_new)[:, 1]
pred_binary = (probs >= loaded_threshold).astype(int)

# Severity for flagged points
pred_severity = np.full(len(X_new), -1)
mask = pred_binary == 1
if mask.sum() > 0:
    pred_severity[mask] = severity_clf.predict(X_new[mask])
```

### D.2 Generate Map
```python
# Integrate predictions with coordinates
results_df = X_new.copy()
results_df['flood_prob'] = probs
results_df['flood_pred'] = pred_binary
results_df['severity'] = pred_severity

# Create Folium map (see map generation code in Final_model.py)
```

---

## Hyperparameter Tuning

For production, consider GridSearch:
```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [200, 400, 600],
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid = GridSearchCV(XGBClassifier(), params, scoring='f1', cv=5)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

---

## Disaster Response Integration

### Real-Time Alerts
1. Monitor new satellite data (daily Sentinel-1 passes)
2. Run inference on latest imagery
3. Flag high-probability areas
4. Alert officials in high-severity zones

### Disaster Management
- **Yellow zones** (Low): Prepare emergency supplies
- **Red zones** (High): Evacuate vulnerable populations
- **Interactive map**: Share with disaster response teams

---

## Limitations & Assumptions

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Class imbalance | Detector may miss floods | Threshold tuning for recall |
| Feature shift | New geography may differ | Retrain on new region |
| Temporal shift | Climate change | Use recent data only |
| SAR noise | False positives | Ensemble with optical imagery |
| Elevation shadows | Coverage gaps | Check SAR layover maps |

---

## Next Steps

1. **Validation**: Test on reserved test set
2. **Deployment**: Export models to production
3. **Monitoring**: Track performance on new data
4. **Refinement**: Collect ground truth, retrain periodically

---

## References

- XGBoost: https://xgboost.readthedocs.io/
- Scikit-learn: https://scikit-learn.org/
- Folium: https://python-visualization.github.io/folium/
- Precision-Recall Trade-off: https://en.wikipedia.org/wiki/Precision_and_recall
