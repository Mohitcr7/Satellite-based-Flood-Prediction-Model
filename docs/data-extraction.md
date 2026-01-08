# Data Extraction Pipeline

## Overview
This module handles the geospatial data extraction workflow. It processes raw satellite and topographic data (KML boundaries, DEM files) to generate a comprehensive feature dataset for flood prediction modeling.

## Purpose
Replaces QGIS manual processing with a pure Python implementation using GDAL, rasterio, richdem, and geopandas libraries. Enables reproducible, scalable terrain analysis.

## Pipeline Steps

### Step 1: KML to GeoJSON Conversion
**File**: `Data_extraction_pipeline.py`  
**Input**: `input.kml` (study area boundary)  
**Output**: `area.geojson`

- Reads KML file using GeoPandas
- Ensures WGS84 (EPSG:4326) coordinate system
- Converts to GeoJSON format for downstream processing

### Step 2: DEM Clipping
**Input**: DEM TIF file (any projection)  
**Output**: `clipped_DEM.tif`

- Clips DEM raster to study area boundary
- Preserves original resolution and projection
- Handles coordinate system transformation automatically

### Step 3: Depression Filling (Hydrological Correction)
**Output**: `clipped_filled_DEM.tif`

- Uses RichDEM library for hydrologically consistent DEM
- Fills local depressions and sinks
- Essential for accurate flow direction analysis
- Equivalent to GRASS `r.fill.dir` command

### Step 4: Terrain Derivatives Generation

#### 4.1 Slope
**Output**: `slope.tif`
- Calculates slope angle (rise/run) from filled DEM
- Units: Dimensionless ratio or degrees
- Range: 0 to 90 degrees

#### 4.2 Aspect
**Output**: `aspect.tif`
- Direction of steepest descent (0-360°)
- 0° = North, 90° = East, 180° = South, 270° = West
- Important for flood flow analysis in mountainous terrain

#### 4.3 Hillshade
**Output**: `hillshade.tif`
- 3D visualization surface using azimuth (315°) and altitude (45°)
- Helps validate DEM quality visually
- Values: 0-255 (8-bit grayscale)

### Step 5: Hydro-Flow Analysis

#### 5.1 Flow Direction
**Output**: `flow_direction.tif`
- D8 algorithm: water flows to steepest downslope neighbor
- 8 possible directions per cell
- Uses pysheds library (GRASS r.watershed equivalent)

#### 5.2 Flow Accumulation
**Output**: `flow_accumulation.tif`
- Cumulative upslope contributing area
- High values = potential stream channels and valleys
- Critical for identifying flood-prone areas
- Units: Number of cells (or area in km²)

### Step 6: Sampling Grid Creation
**Output**: `sampling_grid.gpkg`

- Creates regular point grid at 30m spacing (configurable)
- Points cover entire clipped study area
- Each point has unique ID for tracking

### Step 7: Raster Sampling at Grid Points
**Output**: `sampled_data.csv`

Extracts values from all generated rasters at each grid point:
- `elevation`: DEM elevation value
- `slope`: Slope angle
- `aspect`: Aspect direction
- `hillshade`: Hillshade value (for quality check)
- `flow_accumulation`: Cumulative upstream area
- Geographic coordinates: `longitude`, `latitude` (UTM)

## Input Requirements

| Parameter | Type | Example | Notes |
|-----------|------|---------|-------|
| `kml_path` | String | `/path/input.kml` | Study area boundary polygon |
| `dem_path` | String | `/path/DEM.tif` | Digital Elevation Model |
| `output_dir` | String | `/path/outputs/` | Directory for all outputs |
| Grid spacing | Integer | 30 (meters) | Configurable in code |

## Output CSV Structure

```
id, elevation, slope, aspect, hillshade, flow_accumulation, longitude, latitude
0, 145.23, 0.45, 125, 180, 234.5, 567890, 3456789
1, 147.89, 0.52, 130, 185, 567.2, 567920, 3456819
...
```

## Dependencies

**Required packages**:
```bash
pip install rasterio fiona geopandas shapely richdem numpy pandas scipy
```

**Optional (for advanced flow analysis)**:
```bash
pip install pysheds  # For GRASS-equivalent watershed analysis
```

## Key Parameters

```python
# In Data_extraction_pipeline.py
kml_path = "/path/to/your/input.kml"
dem_path = "/path/to/your/DEM.tif"
output_dir = "/path/to/output/folder"

# Grid spacing (tune for memory/speed tradeoff)
h_spacing = 30  # meters (horizontal)
v_spacing = 30  # meters (vertical)
```

## Performance Notes

- Processing time depends on study area size and grid density
- For 100 km² at 30m spacing: ~20-30 minutes
- Memory requirement: ~2-4 GB for typical DEMs
- Parallelization possible by dividing study area

## Quality Assurance

1. **Visual check**: Use hillshade output to verify DEM quality
2. **Flow validation**: Check flow accumulation (values should follow valley patterns)
3. **Null handling**: Monitor for missing values (gaps indicate processing issues)
4. **Coordinate check**: Verify longitude/latitude range matches study area

## Next Steps

Output CSV from this pipeline (`sampled_data.csv`) feeds into:
→ Data Preprocessing Pipeline (`Data_preprocessing.py`)
→ Combined with flood labels from Google Earth Engine
→ ML-ready dataset for model training

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| KML not found | Invalid path | Check file exists and path is correct |
| CRS mismatch | Different projections | Conversion happens automatically |
| Memory error | Large study area | Reduce grid spacing or divide area |
| NULL values | Data gaps | Check source DEM quality |

## References

- GDAL: https://gdal.org/
- Rasterio: https://rasterio.readthedocs.io/
- RichDEM: https://richdem.readthedocs.io/
- PySheds: https://github.com/mdbartos/pysheds
