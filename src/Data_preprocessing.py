import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("final_dataset_unprocessed.csv")
df2= pd.read_csv("FloodLabels.csv")
df = pd.merge(df, df2, how='inner', on=['X_min','Y_max','X_max','Y_min']) # Merge with flood labels
X = df.drop(columns=['floodClass'])
y= df['floodClass']

# -----------------------------
# 2. DROP METADATA COLUMNS
# -----------------------------
metadata_cols = ['Unnamed: 0', 'index','X_min', 'X_max', 'Y_min', 'Y_max']
X = X.drop(columns=[col for col in metadata_cols if col in X.columns], errors='ignore')

# -----------------------------
# 3. IMPUTE MISSING VALUES
# -----------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])


# -----------------------------
# 4. TRANSFORM SKEWED FEATURES
# -----------------------------
yeo_cols = ["Slope", "Hillshade", "Drainage_direction",
            "Flow_accumulation", "SPI", "TWI"]

pt = PowerTransformer(method='yeo-johnson')
X[yeo_cols] = pt.fit_transform(X[yeo_cols])


log_cols = [ "rainfall_2015", "rainfall_2016", "rainfall_2017", "rainfall_2018",
    "rainfall_2019", "rainfall_2020", "rainfall_2021", "rainfall_2022","rainfall_2023","Distance_to_River"]

for c in log_cols:
    if c in X.columns:
        X[c] = np.log1p(X[c])

# 4.B. NON-LINEAR TRANSFORMATIONS TO CREATE NEW FEATURES
X["Distance_sqrt"] = np.sqrt(np.abs(X["Distance_to_River"]) + 1)
X["Hillshade_pow2"] = X["Hillshade"] ** 2

# -----------------------------
# 5. PCA ON RAINFALL (9 → 2)
# -----------------------------
pca_rain_cols = [
    "rainfall_2015", "rainfall_2016", "rainfall_2017", "rainfall_2018",
    "rainfall_2019", "rainfall_2020", "rainfall_2021", "rainfall_2022",
    "rainfall_2023"
]

pca = PCA(n_components=2)
rain_pca = pca.fit_transform(X[pca_rain_cols])

X["Rainfall_PC1"] = rain_pca[:, 0]
X["Rainfall_PC2"] = rain_pca[:, 1]

X = X.drop(columns=pca_rain_cols)
X = X.drop(columns=['Rainfall_PC2'])  # Drop PC2 to keep only 1 component 

# -----------------------------

# C. INTERACTION FEATURES
X["Rainfall_Elevation_Risk"] = X["rainfall_2024"] * X["Elevation"]
X["Distance_Elevation_Risk"] = X["Distance_sqrt"] / (1 + np.abs(X["Elevation"]))

# E. BINNING — Convert continuous → categorical patterns
X["Hillshade_Category"] = pd.cut(X["Hillshade"], bins=5, labels=False)

# F. DOMAIN-BASED COMPOSITE INDICES
X["flood_risk_index"] = (
    0.35 * X["rainfall_2024"] +
    0.30 * X["Flow_accumulation"] +
    0.20 * (1 / (1 + X["Elevation"])) +
    0.15 * (1 / (1 + X["Distance_sqrt"]))
)

X["Terrain_Drainage_Efficiency"] = np.abs(X["Drainage_direction"] - X["Aspect"])
X["Terrain_Interaction_3Way"] = X["Hillshade"] * X["Aspect"] * X["Elevation"]

# A. DROP REDUNDANT / WEAK FEATURES
drop_cols = ["Slope","SPI",'Distance_to_River']
X = X.drop(columns=drop_cols)
   

# -----------------------------
# 7. STANDARDIZE 
# -----------------------------

num_cols = [c for c in X.select_dtypes(include="number").columns]

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Round scaled columns
X[num_cols] = X[num_cols].round(5)


# -----------------------------
# 9. EXPORT FINAL DATASET
# -----------------------------
df = pd.concat([X, y], axis=1)
df.to_csv("ML_ready_dataset_new.csv", index=False)

print("Processing complete → final_dataset_processed.csv")