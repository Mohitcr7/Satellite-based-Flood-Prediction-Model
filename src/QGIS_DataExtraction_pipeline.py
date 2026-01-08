from qgis.core import QgsProject
import processing
import os

# -----------------------------
# USER INPUTS
# -----------------------------
kml_path = "/path/to/your/input.kml"
dem_path = "/path/to/your/DEM.tif"
output_dir = "/path/to/output/folder"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# STEP 1: Convert KML → GeoJSON
# -----------------------------
geojson_path = os.path.join(output_dir, "area.geojson")
processing.run("native:convertformat", {
    'INPUT': kml_path,
    'TARGET_CRS': 'EPSG:4326',
    'OUTPUT': geojson_path
})
print("✅ KML converted to GeoJSON")

# -----------------------------
# STEP 2: Clip DEM using GeoJSON boundary
# -----------------------------
clipped_dem = os.path.join(output_dir, "clipped_DEM.tif")
processing.run("gdal:cliprasterbymasklayer", {
    'INPUT': dem_path,
    'MASK': geojson_path,
    'SOURCE_CRS': None,
    'TARGET_CRS': None,
    'CROP_TO_CUTLINE': True,
    'KEEP_RESOLUTION': True,
    'DATA_TYPE': 0,
    'OUTPUT': clipped_dem
})
print("✅ DEM clipped successfully")

# -----------------------------
# STEP 3: Fill depressions (Hydrological correction)
# -----------------------------
filled_dem = os.path.join(output_dir, "clipped_filled_DEM.tif")
processing.run("grass7:r.fill.dir", {
    'elevation': clipped_dem,
    'direction': os.path.join(output_dir, "flow_direction_temp.tif"),
    'areas': os.path.join(output_dir, "sink_areas_temp.tif"),
    'GRASS_REGION_PARAMETER': clipped_dem,
    'GRASS_REGION_CELLSIZE_PARAMETER': 0,
    'output': filled_dem,
    'GRASS_OUTPUT_TYPE_PARAMETER': 0
})
print("✅ Depressions filled (Hydrological correction complete)")

# -----------------------------
# STEP 4: Generate Slope, Aspect, Hillshade
# -----------------------------
slope_raster = os.path.join(output_dir, "slope.tif")
aspect_raster = os.path.join(output_dir, "aspect.tif")
hillshade_raster = os.path.join(output_dir, "hillshade.tif")

processing.run("grass7:r.slope.aspect", {
    'elevation': filled_dem,
    'slope': slope_raster,
    'aspect': aspect_raster,
    'pcurvature': None,
    'tcurvature': None,
    'format': 0,
    'precision': 0,
    'zscale': 1.0,
    'min_slope': 0.0,
    'GRASS_REGION_PARAMETER': filled_dem,
    'GRASS_REGION_CELLSIZE_PARAMETER': 0,
    'GRASS_OUTPUT_TYPE_PARAMETER': 0
})
processing.run("gdal:hillshade", {
    'INPUT': filled_dem,
    'Z_FACTOR': 1,
    'AZIMUTH': 315,
    'ALTITUDE': 45,
    'OUTPUT': hillshade_raster
})
print("✅ Generated Slope, Aspect, and Hillshade rasters")

# -----------------------------
# STEP 5: Flow Direction & Water Accumulation
# -----------------------------
flow_dir_raster = os.path.join(output_dir, "flow_direction.tif")
flow_accum_raster = os.path.join(output_dir, "flow_accumulation.tif")

processing.run("grass7:r.watershed", {
    'elevation': filled_dem,
    'accumulation': flow_accum_raster,
    'drainage': flow_dir_raster,
    'threshold': 1000,  # for riverbank flood modeling, set moderate catchment threshold
    'GRASS_REGION_PARAMETER': filled_dem,
    'GRASS_REGION_CELLSIZE_PARAMETER': 0,
    'GRASS_OUTPUT_TYPE_PARAMETER': 0
})
print("✅ Generated Flow Direction and Flow Accumulation rasters")

# -----------------------------
# STEP 6: Create grid matching clipped DEM extent
# -----------------------------
grid_output = os.path.join(output_dir, "sampling_grid.gpkg")

processing.run("native:creategrid", {
    'TYPE': 2,  # point grid
    'EXTENT': clipped_dem,
    'HSPACING': 30,  # 50m spacing — can adjust based on DEM resolution
    'VSPACING': 30,
    'CRS': 'EPSG:4326',
    'OUTPUT': grid_output
})
print("✅ Sampling grid created")

# -----------------------------
# STEP 7: Sample raster values at grid points
# -----------------------------
csv_output = os.path.join(output_dir, "sampled_data.csv")

processing.run("native:rastersampling", {
    'INPUT': grid_output,
    'RASTERS': [filled_dem, slope_raster, aspect_raster, hillshade_raster, flow_accum_raster],
    'COLUMN_PREFIX': 'val_',
    'OUTPUT': csv_output
})
print("✅ Raster values sampled successfully")
print(f"🎯 Data collection pipeline complete. CSV saved at:\n{csv_output}")