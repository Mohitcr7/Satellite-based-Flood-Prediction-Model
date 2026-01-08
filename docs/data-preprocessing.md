# Data Preprocessing Pipeline

## Overview
Transforms raw extracted features into a machine learning-ready dataset. Handles missing values, applies statistical transformations, engineers domain-specific features, and performs dimensionality reduction.

## Purpose
Prepares the unprocessed geospatial data for model training by:
- Handling missing/invalid values
- Normalizing skewed distributions
- Engineering interaction and composite features
- Reducing feature dimensionality
- Standardizing all features for model input

## Input Requirements

| Input | Format | Source | Notes |
|-------|--------|--------|-------|
| **Unprocessed features** | CSV | `Data_extraction_pipeline.py` | Raw geospatial features (elevation, slope, etc.) |
| **Flood labels** | CSV | Google Earth Engine script | Binary/ternary flood classification |
| **Merge key** | Columns | Both datasets | `X_min, Y_max, X_max, Y_min` (grid cell boundaries) |

### Input CSV Structure
```
X_min,Y_max,X_max,Y_min,elevation,slope,aspect,...,rainfall_2015,...
567800,3456900,567830,3456870,145.23,0.45,125,...,850,...
```

## Processing Steps

### Step 1: Data Loading & Merging
```python
df_features = pd.read_csv("final_dataset_unprocessed.csv")  # From extraction pipeline
df_labels = pd.read_csv("FloodLabels.csv")                   # From GEE
df = pd.merge(df_features, df_labels, on=['X_min','Y_max','X_max','Y_min'])
```

**Output**: Unified dataset with features and flood labels

---

### Step 2: Metadata Removal
**Columns dropped**: 
- Index columns: `Unnamed: 0`, `index`
- Grid cell boundaries: `X_min`, `X_max`, `Y_min`, `Y_max`

**Reason**: Metadata adds no predictive signal and prevents model generalization

---

### Step 3: Missing Value Imputation

**Strategy**: Median imputation for numerical columns

```python
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
imputer = SimpleImputer(strategy='median')
X[num_cols] = imputer.fit_transform(X[num_cols])
```

**Why median?** Robust to outliers; preserves distribution better than mean

---

### Step 4: Skewness Reduction via Power Transformation

#### 4.1 Yeo-Johnson Transformation (Box-Cox variant)
Applied to highly skewed terrain features:
- `Slope` - Right-skewed (many flat areas, few steep)
- `Hillshade` - Multi-modal distribution
- `Drainage_direction` - Circular/wrapped data
- `Flow_accumulation` - Exponential (few high values)
- `SPI` (Stream Power Index) - Power-law distribution
- `TWI` (Topographic Wetness Index) - Bimodal

```python
pt = PowerTransformer(method='yeo-johnson')
X[yeo_cols] = pt.fit_transform(X[yeo_cols])
```

#### 4.2 Log Transformation
Applied to strongly right-skewed variables:

```python
log_cols = ["rainfall_2015", ..., "rainfall_2023", "Distance_to_River"]
for col in log_cols:
    X[col] = np.log1p(X[col])  # log1p handles zeros safely
```

**Why log?** Compresses tail, makes distribution more Gaussian for tree-based models

---

### Step 5: Feature Engineering

#### 5.1 Non-linear Transformations
Create new features capturing non-linear relationships:

```python
X["Distance_sqrt"] = np.sqrt(np.abs(X["Distance_to_River"]) + 1)
X["Hillshade_pow2"] = X["Hillshade"] ** 2
```

**Purpose**: Capture quadratic/sqrt relationships missed by linear models

#### 5.2 Dimensionality Reduction: Rainfall PCA
**Problem**: 9 rainfall years (2015-2023) are correlated → redundancy  
**Solution**: PCA reduction from 9 → 2 principal components

```python
pca = PCA(n_components=2)
rain_pca = pca.fit_transform(X[rainfall_cols])
X["Rainfall_PC1"] = rain_pca[:, 0]  # Retained: ~85% variance
# PC2 dropped (low signal)
```

**Result**: 
- `Rainfall_PC1` captures overall precipitation trends
- One fewer feature; cleaner models

#### 5.3 Interaction Features
Capture synergistic effects between variables:

```python
X["Rainfall_Elevation_Risk"] = X["rainfall_2024"] * X["Elevation"]
# Low elevation + high rainfall = higher flood risk
```

#### 5.4 Categorical Binning
Convert continuous to categorical patterns:

```python
X["Hillshade_Category"] = pd.cut(X["Hillshade"], bins=5, labels=False)
# Captures non-linear hillshade-flood relationships
```

#### 5.5 Domain-Based Composite Indices

**Flood Risk Index** (weighted combination):
```python
X["flood_risk_index"] = (
    0.35 * X["rainfall_2024"] +        # Rainfall weight: 35%
    0.30 * X["Flow_accumulation"] +    # Flow weight: 30%
    0.20 * (1/(1 + X["Elevation"])) +  # Elevation (inverse): 20%
    0.15 * (1/(1 + X["Distance_sqrt"]))# Distance (inverse): 15%
)
```
**Rationale**: High rainfall + high flow + low elevation + close to river = high flood risk

**Terrain Drainage Efficiency**:
```python
X["Terrain_Drainage_Efficiency"] = np.abs(X["Drainage_direction"] - X["Aspect"])
# Mismatch between drainage and aspect indicates flow obstacles
```

**3-Way Terrain Interaction**:
```python
X["Terrain_Interaction_3Way"] = X["Hillshade"] * X["Aspect"] * X["Elevation"]
# Captures complex terrain interactions
```

---

### Step 6: Feature Dropping (Redundancy Removal)

**Columns dropped**:
- `Slope` - Redundant with flow direction; high correlation
- `SPI` - Highly specific; correlation with flow_accumulation
- `Distance_to_River` - Original distance column (replaced by `Distance_sqrt` and composite indices)

**Final feature count**: Typically 25-35 features (down from initial 50+)

---

### Step 7: Standardization & Normalization

**Method**: StandardScaler (z-score normalization)

```python
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
# Output: mean=0, std=1 for all numerical features
```

**Why standardize?**
- Required by many ML algorithms (KNN, SVM, Neural Networks)
- Prevents high-magnitude features from dominating
- Improves model convergence speed

**Rounding**: Scaled values rounded to 5 decimal places (numerical stability)

---

### Step 8: Final Dataset Assembly

```python
df_final = pd.concat([X_processed, y], axis=1)
df_final.to_csv("ML_ready_dataset.csv", index=False)
```

**Output CSV structure**:
```
elevation, slope_transformed, aspect, ..., flood_risk_index, Rainfall_PC1, floodClass
-0.45201,  0.32145,         -0.891, ..., 0.567,            -1.234,       1
 0.12345,  -0.15234,         0.678, ..., -0.234,            0.456,        0
```

---

## Feature Descriptions

| Feature | Type | Range | Domain Role |
|---------|------|-------|-------------|
| `elevation` | Numerical | Transformed | Base terrain |
| `slope_transformed` | Numerical | Transformed | Gradient/steepness |
| `flow_accumulation` | Numerical | Transformed | Drainage intensity |
| `rainfall_*` | Numerical | Log-transformed | Precipitation input |
| `Rainfall_PC1` | Numerical | [-3, 3] | Dimensionally reduced rainfall |
| `flood_risk_index` | Numerical | Scaled | Composite flood indicator |
| `floodClass` | Categorical | {0, 1, 2} | **Target**: 0=No Flood, 1=Low, 2=High |

---

## Dependencies

```bash
pip install pandas numpy scikit-learn
```

**Key modules**:
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `sklearn.preprocessing`: Scaling, transformation
- `sklearn.decomposition`: PCA

---

## Configuration Parameters

```python
# Input/output paths
INPUT_UNPROCESSED = "final_dataset_unprocessed.csv"
INPUT_LABELS = "FloodLabels.csv"
OUTPUT_ML_READY = "ML_ready_dataset.csv"

# Transformation parameters
RANDOM_STATE = 42
PCA_COMPONENTS = 2  # For rainfall
STANDARDIZER = StandardScaler()

# Feature engineering weights
RAINFALL_WEIGHT = 0.35
FLOW_WEIGHT = 0.30
ELEVATION_WEIGHT = 0.20
DISTANCE_WEIGHT = 0.15
```

---

## Output Quality Metrics

Check after processing:
```python
print("Dataset shape:", df_final.shape)           # (N_samples, N_features)
print("Missing values:", df_final.isnull().sum()) # Should be 0
print("Feature stats:\n", df_final.describe())    # Mean ~0, std ~1
print("Class distribution:\n", df_final['floodClass'].value_counts())
```

---

## Next Steps

ML-ready dataset (`ML_ready_dataset.csv`) feeds into:
→ Model Training Pipeline (`Final_model.py`)
→ Binary flood detector (Yes/No)
→ Flood severity classifier (Low/High)
→ Interactive mapping & predictions

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Shape mismatch in merge | Different number of grid cells | Verify extraction pipeline output |
| NaN after transform | Log of negative/zero values | Use `np.log1p()` instead of `np.log()` |
| Standardization failure | Non-numeric columns present | Verify metadata removed in Step 2 |
| Class imbalance | Unequal flood/non-flood samples | Handled by model (SMOTE in training) |
