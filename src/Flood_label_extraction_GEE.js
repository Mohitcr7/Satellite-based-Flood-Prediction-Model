// ------------------------------------------------------
//              USER INPUTS
// ------------------------------------------------------
var points = ee.FeatureCollection("projects/weighty-time-471514-b0/assets/grid_points_WGS84");

// Flood event window (example)
var before_date = "2023-07-01";
var after_date  = "2023-07-15";

// Buffer for sampling (30 m recommended)
var BUFFER_METERS = 30;

// SAR water threshold (Himalayan rivers ~ -12 to -14 dB)
var WATER_THRESHOLD = -13;   // dB



// ------------------------------------------------------
//          LOAD SENTINEL-1 BEFORE & AFTER IMAGES
// ------------------------------------------------------
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

var before = getVV(before_date);
var after  = getVV(after_date);

print("Before SAR:", before);
print("After SAR:", after);



// ------------------------------------------------------
//                 FLOOD FEATURE EXTRACTION
// ------------------------------------------------------
var extractFlood = function(pt) {

  var geom = pt.geometry().buffer(BUFFER_METERS);

  // Sample before & after VV values
  var b = before.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: 10,
    maxPixels: 1e9
  }).get('VV');

  var a = after.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: 10,
    maxPixels: 1e9
  }).get('VV');

  // Replace NULL with realistic SAR value (−20 dB)
  var vv_b = ee.Number(ee.Algorithms.If(b, b, -20));
  var vv_a = ee.Number(ee.Algorithms.If(a, a, -20));

  // SAR flood indicator = drop in VV
  var diff = vv_b.subtract(vv_a);

  // ------------------------------------------------------
  //      Water Fraction from post-flood SAR
  // ------------------------------------------------------
  var waterMask = after.lt(WATER_THRESHOLD).rename('water');

  var waterFraction = waterMask.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: 10,
    maxPixels: 1e9
  }).get('water');

  var frac = ee.Number(ee.Algorithms.If(waterFraction, waterFraction, 0));


  // ------------------------------------------------------
  //                FLOOD CLASSIFICATION
  // ------------------------------------------------------
  var isHigh = diff.gte(2.5).or(frac.gt(0.40));
  var isMed  = diff.gte(1.2).or(frac.gt(0.15));

  var floodClass = ee.Number(
    ee.Algorithms.If(isHigh, 2,
      ee.Algorithms.If(isMed, 1, 0))
  );

  // Debug print (optional)
  // print("Debug:", pt.id(), vv_b, vv_a, diff, frac, floodClass);

  return pt.set({
    VV_before: vv_b,
    VV_after: vv_a,
    VV_change: diff,
    water_fraction: frac,
    floodClass: floodClass,
    event_date_before: before_date,
    event_date_after: after_date
  });
};



// ------------------------------------------------------
//                    APPLY TO ALL POINTS
// ------------------------------------------------------
var floodPoints = points.map(extractFlood);
print("Final Labeled Dataset:", floodPoints.limit(5));



// ------------------------------------------------------
//                    EXPORT CSV
// ------------------------------------------------------
Export.table.toDrive({
  collection: floodPoints,
  description: "FloodLabels_Final",
  fileFormat: "CSV"
});