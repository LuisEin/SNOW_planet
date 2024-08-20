from osgeo import gdal, ogr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

# ====================
# Define Your Paths Here
# ====================
# Path to the training data GeoTIFF (Snow=1, NoSnow=0, NoData=255)
training_data_tif = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/training_data/20240229_binary_snow_training_CBSI_non_shaded.tif'

# Path to the indexed scene that needs to be classified
path_to_index_tif = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shade_no_shade_8b_TOAR_gaussian_filtered/non_shaded_offset_0.02/20240229_101353_PS_TOAR_8b_gaussian_TOAR_CBSI_width_3334px_without_shadow.tif'

# Path to the shaded area mask
path_to_shaded_mask = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/mask_files_gaussian_filtered_offset_0.02/20240229_101353_PS_TOAR_8b_shadow_masked_gaussian_filtered.tif'

# Path to save the Random Forest model trained on the index
model_save_path_index = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/models/non_shaded/20240229_rf_model_index.pkl'

# Path to save the predicted snow map using the index model
predicted_snow_map_index_tmp = '/tmp/20240229_predicted_snow_map_index_non_shaded.tif'

# Path to the true extent raster file (contains NoData for uncovered areas and Eibsee)
true_extent_raster_path = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band_gaussian_filtered/20240229_101353_PS_TOAR_8b_gaussian.tif'

# Path to the AOI shapefile
aoi_shapefile = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Shapefiles/shapefile_Zugspitze/03_AOI_shp_zugspitze_reproj_for_code/AOI_zugspitze_reproj_32632.shp'

# Path to save the final masked snow map
output_masked_snow_map_path = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/binary_classification/non_shaded/20240229_masked_snow_map_index_non_shaded.tif'

# ====================
# Functions
# ====================

def load_raster(tif_path):
    """Load a raster and return its array, geotransform, and projection."""
    dataset = gdal.Open(tif_path)
    band = dataset.GetRasterBand(1).ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    nodata_value = dataset.GetRasterBand(1).GetNoDataValue()
    return band, geotransform, projection, nodata_value

def resample_raster_to_match(reference_raster_path, target_raster_path, output_raster_path):
    """Resample the target raster to match the reference raster."""
    reference_raster = gdal.Open(reference_raster_path)
    reference_proj = reference_raster.GetProjection()
    reference_geotransform = reference_raster.GetGeoTransform()
    reference_x_size = reference_raster.RasterXSize
    reference_y_size = reference_raster.RasterYSize

    target_raster = gdal.Open(target_raster_path)

    resampled_raster = gdal.Warp(
        output_raster_path,
        target_raster,
        format='GTiff',
        width=reference_x_size,
        height=reference_y_size,
        dstSRS=reference_proj,
        outputBounds=(
            reference_geotransform[0],
            reference_geotransform[3] + reference_y_size * reference_geotransform[5],
            reference_geotransform[0] + reference_x_size * reference_geotransform[1],
            reference_geotransform[3]
        ),
        resampleAlg=gdal.GRA_NearestNeighbour
    )

    return output_raster_path

def create_training_data_from_raster(training_raster, index_raster, nodata_value=255):
    """Create a DataFrame with training data extracted from the training raster and index raster."""
    valid_mask = (training_raster != nodata_value) & (index_raster != nodata_value)
    data = {
        'index': index_raster[valid_mask].flatten(),
        'label': training_raster[valid_mask].flatten()
    }
    df = pd.DataFrame(data)
    
    if df['label'].nunique() < 2:
        raise ValueError("Training data does not contain both classes (1 and 0). Check your input data.")
    
    return df

def train_model(training_data, model_save_path):
    """Train the Random Forest model."""
    X = training_data[['index']]
    y = training_data['label']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, model_save_path)

def apply_shaded_mask_to_prediction(prediction_raster, shaded_mask_raster, nodata_value):
    """Apply the shaded mask to the prediction raster."""
    shaded_mask = shaded_mask_raster == 1  # Areas with 1 in the shaded mask are shaded areas
    
    # Ensure nodata_value is an integer
    if np.isnan(nodata_value):
        nodata_value = 255  # Assuming 255 is appropriate for your data type
    
    prediction_raster[shaded_mask] = nodata_value  # Set shaded areas to NoData value
    return prediction_raster

def run_prediction(input_raster, model_path, output_path, geotransform, projection, nodata_value, shaded_mask_path):
    """Run prediction on the input raster using the trained model and save the output, applying shaded mask."""
    model = joblib.load(model_path)
    
    # Flatten the input data and create a mask for valid data
    input_flat = input_raster.flatten()

    # Ensure nodata_value is an integer
    nodata_value = int(nodata_value) if not np.isnan(nodata_value) else 255
    
    # Preserve NoData values and prepare the input data for prediction
    valid_mask = input_flat != nodata_value
    input_data = pd.DataFrame({'index': input_flat[valid_mask]})
    
    # Predict only for valid data
    predictions = np.full_like(input_flat, nodata_value, dtype=np.uint8)
    predictions[valid_mask] = model.predict(input_data)
    
    # Reshape the predictions to the original raster shape
    predictions = predictions.reshape(input_raster.shape)
    
    # Apply shaded mask to predictions
    shaded_mask_raster, _, _, _ = load_raster(shaded_mask_path)
    predictions = apply_shaded_mask_to_prediction(predictions, shaded_mask_raster, nodata_value)

    # Save the prediction raster
    driver = gdal.GetDriverByName('GTiff')
    outdata = driver.Create(output_path, predictions.shape[1], predictions.shape[0], 1, gdal.GDT_Byte)
    outdata.SetGeoTransform(geotransform)
    outdata.SetProjection(projection)
    outdata.GetRasterBand(1).WriteArray(predictions)
    outdata.GetRasterBand(1).SetNoDataValue(nodata_value)
    outdata.FlushCache()

# ====================
# Main Workflow
# ====================

# Load the index file (this will define the output size and position)
index_raster, geotransform, projection, index_nodata_value = load_raster(path_to_index_tif)

# Resample the training data to match the index file
resampled_training_raster_path = '/tmp/resampled_training_raster.tif'
resample_raster_to_match(path_to_index_tif, training_data_tif, resampled_training_raster_path)

# Load the resampled training data
training_raster, _, _, training_nodata_value = load_raster(resampled_training_raster_path)

# Create training data
training_data = create_training_data_from_raster(training_raster, index_raster, nodata_value=training_nodata_value)

# Train the model
train_model(training_data, model_save_path_index)

# Run predictions using the index file as reference and apply the shaded mask
run_prediction(
    index_raster,
    model_save_path_index,
    predicted_snow_map_index_tmp,
    geotransform,
    projection,
    index_nodata_value,
    path_to_shaded_mask
)

# ====================
# Apply Final Masking
# ====================
# Load the prediction raster from the temporary location
dataset = gdal.Open(predicted_snow_map_index_tmp)
prediction_raster = dataset.GetRasterBand(1).ReadAsArray()
geotransform = dataset.GetGeoTransform()
projection = dataset.GetProjection()
nodata_value = dataset.GetRasterBand(1).GetNoDataValue()

# Load the true extent raster
true_extent_dataset = gdal.Open(true_extent_raster_path)
true_extent_raster = true_extent_dataset.GetRasterBand(1).ReadAsArray()
true_extent_nodata_value = true_extent_dataset.GetRasterBand(1).GetNoDataValue()

# Create a mask for the AOI
aoi_ds = ogr.Open(aoi_shapefile)
aoi_layer = aoi_ds.GetLayer()

# Create an in-memory raster to burn the AOI shapefile into
aoi_mask_ds = gdal.GetDriverByName('MEM').Create(
    '', prediction_raster.shape[1], prediction_raster.shape[0], 1, gdal.GDT_Byte)
aoi_mask_ds.SetGeoTransform(geotransform)
aoi_mask_ds.SetProjection(projection)
aoi_band = aoi_mask_ds.GetRasterBand(1)
aoi_band.Fill(0)  # Initialize with 0s
aoi_band.SetNoDataValue(nodata_value)

# Burn the AOI into the mask
gdal.RasterizeLayer(aoi_mask_ds, [1], aoi_layer, burn_values=[1])

# Get the AOI mask array
aoi_mask = aoi_band.ReadAsArray()

# Create a mask for the true extent raster's coverage
true_extent_mask = (true_extent_raster != true_extent_nodata_value)

# Identify areas within the AOI but not covered by the true extent raster
uncovered_mask = (aoi_mask == 1) & (~true_extent_mask)

# Ensure that prediction_raster is of a type that can hold the nodata_value
if np.issubdtype(prediction_raster.dtype, np.integer):
    if nodata_value > np.iinfo(prediction_raster.dtype).max:
        prediction_raster = prediction_raster.astype(np.uint8)
elif np.issubdtype(prediction_raster.dtype, np.floating):
    if nodata_value > np.finfo(prediction_raster.dtype).max:
        prediction_raster = prediction_raster.astype(np.float32)

# Apply the uncovered_mask to the prediction_raster
prediction_raster[uncovered_mask == 1] = nodata_value

# Save the final masked raster
driver = gdal.GetDriverByName('GTiff')
outdata = driver.Create(output_masked_snow_map_path, prediction_raster.shape[1], prediction_raster.shape[0], 1, gdal.GDT_Byte)
outdata.SetGeoTransform(geotransform)
outdata.SetProjection(projection)
outdata.GetRasterBand(1).WriteArray(prediction_raster)
outdata.GetRasterBand(1).SetNoDataValue(nodata_value)
outdata.FlushCache()

print(f'Masked snow map saved to {output_masked_snow_map_path}')

# Optional: Plot the results
plt.figure(figsize=(10, 8))
plt.title("Uncovered Mask")
plt.imshow(uncovered_mask, cmap='gray')
plt.colorbar(label='Mask Value')
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Prediction Raster")
plt.imshow(prediction_raster, cmap='gray')
plt.colorbar(label='Prediction Value')
plt.show()
