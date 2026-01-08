# Google Earth Engine Flood Label Extraction

## Overview
Uses Sentinel-1 synthetic aperture radar (SAR) satellite imagery to automatically detect and classify flood events. Exports flood labels for the study area to be merged with geospatial features.

## Purpose
Eliminates manual flood labeling by leveraging SAR's all-weather, day/night imaging capability to detect surface water changes (floods) during specific events.

## Key Concepts

### Synthetic Aperture Radar (SAR)
- **Instrument**: Sentinel-1 C-band SAR (COPERNICUS/S1_GRD)
- **Wavelength**: 5.6 cm (C-band)
- **Advantage**: Works through clouds and darkness (ideal for monsoon floods)
- **Measurement**: Backscatter intensity (how much signal reflects back)

### Water Detection Principle
Water surfaces are **smooth** → act as mirrors → low backscatter (dark in SAR)  
Land surfaces are **rough** → scatter signal → high backscatter (bright in SAR)

```
Dry land:     ██████ (high backscatter, e.g., -10 dB)
Flooded area: ░░░░░░ (low backscatter, e.g., -18 dB)
```

---

## Input Configuration

```javascript
// Study area grid points
var points = ee.FeatureCollection("projects/weighty-time-471514-b0/assets/grid_points_WGS84");

// Flood event dates
var before_date = "2023-07-01";  // Pre-flood baseline
var after_date = "2023-07-15";   // During/post-flood

// Spatial buffer around each point
var BUFFER_METERS = 30;

// SAR water threshold
var WATER_THRESHOLD = -13;  // dB (Himalayan rivers: -12 to -14 dB)
```

### Input Requirements

| Parameter | Value | Notes |
|-----------|-------|-------|
| **points** | GEE FeatureCollection | Grid points from extraction pipeline |
| **before_date** | YYYY-MM-DD | 2-4 weeks before flood event |
| **after_date** | YYYY-MM-DD | During or after peak flood |
| **BUFFER_METERS** | Integer (30) | Spatial buffer for sampling (same as grid spacing) |
| **WATER_THRESHOLD** | Decimal dB | Tuned per region; Himalaya: -13 dB |

---

## Processing Pipeline

### Step 1: SAR Image Collection (Before Event)
```javascript
function getVV(date) {
  return ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(points)
    .filterDate(date, ee.Date(date).advance(5, 'day'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .select('VV')
    .mean()
    .clip(points);
}
```

**Filters**:
- **Bounds**: Spatial coverage of study area
- **Date**: Within 5-day window
- **Mode**: IW (Interferometric Wide) - standard mode
- **Pass**: DESCENDING - consistent geometry
- **Resolution**: 10m pixels
- **Polarization**: VV (vertical transmit, vertical receive) - sensitive to water

**Output**: Mean VV backscatter image for pre-flood period

---

### Step 2: SAR Change Detection
```javascript
var before = getVV(before_date);
var after = getVV(after_date);

var vv_b = ee.Number(ee.Algorithms.If(b, b, -20));  // Before: -20 dB default
var vv_a = ee.Number(ee.Algorithms.If(a, a, -20));  // After: -20 dB default

var diff = vv_b.subtract(vv_a);  // Positive = backscatter drop
```

**Interpretation**:
- `diff > 0`: Backscatter decreased (water likely)
- `diff = 0`: No change (no flood)
- `diff < 0`: Backscatter increased (unlikely for floods)

---

### Step 3: Water Fraction Calculation
```javascript
var waterMask = after.lt(WATER_THRESHOLD);  // Pixels darker than threshold
var waterFraction = waterMask.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: geom,
  scale: 10,
  maxPixels: 1e9
}).get('water');
```

**Result**: Proportion of pixels meeting water criterion (0.0 to 1.0)

- `waterFraction = 0.5`: 50% of sampled area is water
- `waterFraction = 1.0`: 100% water (fully flooded)

---

### Step 4: Hierarchical Flood Classification

```javascript
var isHigh = diff.gte(2.5).or(frac.gt(0.40));
var isMed  = diff.gte(1.2).or(frac.gt(0.15));

var floodClass = ee.Number(
  ee.Algorithms.If(isHigh, 2,
  ee.Algorithms.If(isMed, 1, 0))
);
```

#### Classification Logic

| Condition | Class | Interpretation |
|-----------|-------|-----------------|
| `diff >= 2.5` **OR** `frac > 0.40` | 2 = **HIGH** | Severe flooding |
| `diff >= 1.2` **OR** `frac > 0.15` | 1 = **LOW** | Mild/moderate flooding |
| Otherwise | 0 = **NO FLOOD** | No water change |

**Thresholds** (tuned for Himalayan region):
- **Backscatter change threshold**: 2.5 dB (high), 1.2 dB (medium)
- **Water fraction threshold**: 40% (high), 15% (low)

---

### Step 5: Feature Extraction per Point

For each grid point, script exports:

```javascript
return pt.set({
  VV_before: vv_b,              // Pre-flood SAR value (dB)
  VV_after: vv_a,               // Post-flood SAR value (dB)
  VV_change: diff,              // Backscatter change (dB)
  water_fraction: frac,         // Water proportion (0-1)
  floodClass: floodClass,       // Final label (0/1/2)
  event_date_before: before_date,
  event_date_after: after_date
});
```

---

## Output Format

### Export to Google Drive
```javascript
Export.table.toDrive({
  collection: floodPoints,
  description: "FloodLabels_Final",
  fileFormat: "CSV"
});
```

### Output CSV Structure
```csv
X_min,Y_max,X_max,Y_min,VV_before,VV_after,VV_change,water_fraction,floodClass
567800,3456900,567830,3456870,-15.2,-17.8,-2.6,0.35,1
567830,3456900,567860,3456870,-14.8,-18.5,-3.7,0.52,2
...
```

---

## Merge with Features

**Keys for merging**:
- `X_min, Y_max, X_max, Y_min` (grid cell boundaries)

```python
# In Data_preprocessing.py
df_features = pd.read_csv("final_dataset_unprocessed.csv")  # From extraction
df_labels = pd.read_csv("FloodLabels.csv")                  # From GEE
df_merged = pd.merge(df_features, df_labels, 
                     on=['X_min','Y_max','X_max','Y_min'])
```

**Result**: Each feature row now has corresponding flood label

---

## Tuning Parameters by Region

| Region | WATER_THRESHOLD | Rationale |
|--------|-----------------|-----------|
| **Himalayan rivers** | -13 dB | Clear mountain streams |
| **Large rivers** | -15 dB | Rougher water surfaces |
| **Urban flooded areas** | -10 dB | Buildings/infrastructure scatter signal |
| **Agricultural land** | -14 dB | Seasonal vegetation affects backscatter |

**Testing approach**:
1. Run script with baseline threshold
2. Visual inspection of results in Earth Engine
3. Compare with ground truth (if available)
4. Adjust by ±1 dB and re-run

---

## How to Use

### In Google Earth Engine Code Editor:

1. **Copy script** into Code Editor (https://code.earthengine.google.com/)
2. **Update parameters**:
   ```javascript
   var points = ee.FeatureCollection("YOUR_ASSET_PATH");
   var before_date = "YOUR_DATE_1";
   var after_date = "YOUR_DATE_2";
   var WATER_THRESHOLD = YOUR_VALUE;
   ```
3. **Run script** (▶ button)
4. **Check console** for sample outputs (10 rows printed)
5. **Start export** (Run → Tasks → Run)
6. **Wait for completion** (check Drive)
7. **Download CSV** from Google Drive

---

## Debugging & Validation

### Print Sample Data
```javascript
print("Sample labeled points:", floodPoints.limit(5));
```
Shows first 5 rows in console

### Visualize SAR Images
```javascript
Map.addLayer(before, {min: -30, max: 0}, 'Before SAR');
Map.addLayer(after, {min: -30, max: 0}, 'After SAR');
```
Allows visual inspection of changes

### Check Class Distribution
```javascript
print("Flood classes:", floodPoints.aggregate_histogram('floodClass'));
```
Returns: `{0: 450, 1: 89, 2: 12}` (example)

---

## Limitations & Considerations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Cloud cover | Noise in VV values | Averages over 5-day window |
| Urban areas | High backscatter (confusing) | Exclude from study area if possible |
| Dense vegetation | Masks water signal | Use descending pass (less sensitive to vegetation) |
| Layover/shadow | False negatives | Check DEM-corrected SAR |
| SAR data gaps | Missing labels | Choose dates with guaranteed coverage |

---

## Next Steps

Labeled CSV (`FloodLabels.csv`) merges with:
→ Features from Data Extraction Pipeline
→ Data Preprocessing Pipeline
→ ML Training Pipeline with both features and labels

---

## References

- Sentinel-1 Documentation: https://sentinel.esa.int/web/sentinel/missions/sentinel-1
- SAR Flood Detection: https://www.earthobservations.org/
- Google Earth Engine: https://earthengine.google.com/
