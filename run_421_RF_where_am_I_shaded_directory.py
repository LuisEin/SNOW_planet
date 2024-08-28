#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:21:45 2024

This script processes a directory of Coastal Blue (CB) TIFF files to classify 
Snow coverage in shaded areas using a Random Forest model.
It matches files by date across multiple directories, trains a model 
For each date, applies masks to make sure only shaded areas contain valid data, 
And saves the results.

@author: luis
"""

from osgeo import gdal, ogr
import numpy as np
import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# ====================
# Define Your Paths Here
# ====================
training_data_tif = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/training_data/shaded/20240229_binary_snow_training_Coastal_Blue_shaded_87_Polygons.tif'
CB_tif_dir ='/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shadow_masked_original_ready_files_gaussian_filtered_offset_0.02/*.tif'
path_to_non_shaded_mask_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shade_no_shade_8b_TOAR_gaussian_filtered/non_shaded_offset_0.02/*.tif'
non_shaded_mask_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/non-shaded_mask/mask_files_gaussian_filtered_offset_0.02'
true_extent_raster_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band_gaussian_filtered/'
aoi_shapefile = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Shapefiles/shapefile_Zugspitze/03_AOI_shp_zugspitze_reproj_for_code/AOI_zugspitze_reproj_32632.shp'
model_save_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/models/shaded/'
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/binary_classification/shaded/'

# ====================
# Functions
# ====================

def load_raster(tif_path, band_number=1):
    """Load a raster and return its array, geotransform, projection, and NoData value."""
    dataset = gdal.Open(tif_path)
    band = dataset.GetRasterBand(band_number).ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    nodata_value = dataset.GetRasterBand(band_number).GetNoDataValue()
    return band, geotransform, projection, nodata_value

def resample_raster(reference_raster_path, target_raster_path, output_raster_path):
    """Resample the target raster to match the reference raster."""
    reference_raster = gdal.Open(reference_raster_path)
    reference_proj = reference_raster.GetProjection()
    reference_geotransform = reference_raster.GetGeoTransform()
    reference_x_size = reference_raster.RasterXSize
    reference_y_size = reference_raster.RasterYSize

    target_raster = gdal.Open(target_raster_path)

    gdal.Warp(
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

def create_training_data(training_raster, index_raster, training_nodata_value, index_nodata_value):
    """Create a DataFrame with training data from the training and index rasters."""
    valid_mask = (training_raster != training_nodata_value) & (index_raster != index_nodata_value)
    data = {
        'index': index_raster[valid_mask].flatten(),
        'label': training_raster[valid_mask].flatten()
    }
    training_data = pd.DataFrame(data)

    if training_data['label'].nunique() < 2:
        raise ValueError("Training data does not contain both classes (1 and 0). Check your input data.")

    return training_data

def train_rf_model(training_data, model_save_path):
    """Train a Random Forest model and save it."""
    X = training_data[['index']]
    y = training_data['label']

    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_save_path)
    return model

def create_binary_mask_tif(shady_file, mask_dir):
    """Create a binary mask from the shaded TIFF file."""
    ds = gdal.Open(shady_file)
    if ds is None:
        raise FileNotFoundError(f"Failed to open {shady_file}")

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()

    if data is None:
        raise ValueError(f"No valid data in {shady_file}")

    mask = np.where(np.isnan(data), 0, 1)

    base_name = os.path.basename(shady_file).split('_PS_TOAR_8b')[0]
    output_mask_path = os.path.join(mask_dir, f"{base_name}_PS_TOAR_8b_non_shaded_mask.tif")

    driver = gdal.GetDriverByName("GTiff")
    mask_ds = driver.Create(
        output_mask_path,
        ds.RasterXSize,
        ds.RasterYSize,
        1,
        gdal.GDT_Float32,
    )
    mask_ds.SetGeoTransform(ds.GetGeoTransform())
    mask_ds.SetProjection(ds.GetProjection())
    mask_ds.GetRasterBand(1).WriteArray(mask)

    mask_ds = None
    ds = None

    return output_mask_path

def apply_non_shaded_mask(predictions, non_shaded_mask, nodata_value, geotransform, projection):
    """Apply the non-shaded mask to the prediction raster."""
    # Resample non_shaded_mask to match the dimensions of predictions if needed
    if non_shaded_mask.shape != predictions.shape:
        # Create an in-memory raster for resampling
        mem_driver = gdal.GetDriverByName('MEM')
        mem_raster = mem_driver.Create('', predictions.shape[1], predictions.shape[0], 1, gdal.GDT_Float32)
        mem_raster.SetGeoTransform(geotransform)
        mem_raster.SetProjection(projection)
        gdal.ReprojectImage(gdal.Open(non_shaded_mask_path), mem_raster, None, None, gdal.GRA_NearestNeighbour)
        
        # Read the resampled mask
        non_shaded_mask = mem_raster.GetRasterBand(1).ReadAsArray()

    predictions[non_shaded_mask == 1] = nodata_value
    return predictions

def apply_aoi_and_extent_masks(predictions, geotransform, projection, index_nodata_value, true_extent_raster_path, aoi_shapefile):
    """Apply AOI and true extent masks to the predictions."""
    true_extent_dataset = gdal.Open(true_extent_raster_path)
    true_extent_raster = true_extent_dataset.GetRasterBand(1).ReadAsArray()
    true_extent_nodata_value = true_extent_dataset.GetRasterBand(1).GetNoDataValue()

    aoi_ds = ogr.Open(aoi_shapefile)
    aoi_layer = aoi_ds.GetLayer()

    aoi_mask_ds = gdal.GetDriverByName('MEM').Create(
        '', predictions.shape[1], predictions.shape[0], 1, gdal.GDT_Byte)
    aoi_mask_ds.SetGeoTransform(geotransform)
    aoi_mask_ds.SetProjection(projection)
    aoi_band = aoi_mask_ds.GetRasterBand(1)
    aoi_band.Fill(0)
    aoi_band.SetNoDataValue(index_nodata_value)

    gdal.RasterizeLayer(aoi_mask_ds, [1], aoi_layer, burn_values=[1])

    aoi_mask = aoi_band.ReadAsArray()

    true_extent_mask = (true_extent_raster != true_extent_nodata_value)

    uncovered_mask = (aoi_mask == 1) & (~true_extent_mask)

    if np.issubdtype(predictions.dtype, np.integer):
        if index_nodata_value > np.iinfo(predictions.dtype).max:
            predictions = predictions.astype(np.uint8)
    elif np.issubdtype(predictions.dtype, np.floating):
        if index_nodata_value > np.finfo(predictions.dtype).max:
            predictions = predictions.astype(np.float32)

    predictions[uncovered_mask == 1] = index_nodata_value

    return predictions

def save_raster(predictions, output_path, geotransform, projection, nodata_value):
    """Save the predictions raster to a file."""
    driver = gdal.GetDriverByName('GTiff')
    outdata = driver.Create(output_path, predictions.shape[1], predictions.shape[0], 1, gdal.GDT_Byte)
    outdata.SetGeoTransform(geotransform)
    outdata.SetProjection(projection)
    outdata.GetRasterBand(1).WriteArray(predictions)
    outdata.GetRasterBand(1).SetNoDataValue(nodata_value)
    outdata.FlushCache()

    print(f'Masked snow map saved to {output_path}')

# ====================
# Main Workflow
# ====================

def process_file(path_to_CB_tif, path_to_non_shaded_mask, true_extent_raster_path, output_masked_snow_map_path, model_save_path_index):
    # Load the index file
    index_raster, geotransform, projection, index_nodata_value = load_raster(path_to_CB_tif)

    # Resample the training data
    resampled_training_raster_path = resample_raster(path_to_CB_tif, training_data_tif, '/tmp/resampled_training_raster_shaded.tif')

    # Load the resampled training data
    training_raster, _, _, training_nodata_value = load_raster(resampled_training_raster_path)

    # Create training data
    training_data = create_training_data(training_raster, index_raster, training_nodata_value, index_nodata_value)

    # Train the model independently for this image
    model = train_rf_model(training_data, model_save_path_index)

    # Run predictions
    input_flat = index_raster.flatten()
    if index_nodata_value is None or (isinstance(index_nodata_value, float) and np.isnan(index_nodata_value)):
        index_nodata_value = 255  # Default NoData value if original is None or NaN

    valid_mask = input_flat != index_nodata_value
    input_data = pd.DataFrame({'index': input_flat[valid_mask]})

    # Predict for this image using the trained model
    predictions = np.full_like(input_flat, index_nodata_value, dtype=np.uint8)
    predictions[valid_mask] = model.predict(input_data)
    predictions = predictions.reshape(index_raster.shape)

    # Create non-shaded mask
    non_shaded_mask_path = create_binary_mask_tif(path_to_non_shaded_mask, non_shaded_mask_dir)

    # Load the created non-shaded mask
    non_shaded_mask, _, _, _ = load_raster(non_shaded_mask_path)

    # Apply non-shaded mask to predictions
    predictions = apply_non_shaded_mask(predictions, non_shaded_mask, index_nodata_value, geotransform, projection)

    # Apply AOI and true extent masks to predictions
    predictions = apply_aoi_and_extent_masks(predictions, geotransform, projection, index_nodata_value, true_extent_raster_path, aoi_shapefile)

    # Save the prediction raster
    save_raster(predictions, output_masked_snow_map_path, geotransform, projection, index_nodata_value)

    # Clean up any temporary data related to this image
    os.remove(resampled_training_raster_path)

def main():
    # Get list of all Coastal Blue TIFF files and sort them by date
    CB_tif_files = sorted(glob.glob(CB_tif_dir))

    # Sort non-shaded mask files by date
    non_shaded_mask_files = sorted(glob.glob(path_to_non_shaded_mask_dir))

    # Ensure the directories for models and output exist
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Process files based on matching dates
    for CB_tif_file in CB_tif_files:
        base_name = os.path.basename(CB_tif_file).split('_PS_TOAR_8b')[0]

        # Find the matching non-shaded mask file
        matching_non_shaded_mask_file = [f for f in non_shaded_mask_files if base_name in os.path.basename(f)]
        if not matching_non_shaded_mask_file:
            print(f"No matching non-shaded mask found for {CB_tif_file}")
            continue
        non_shaded_mask_file = matching_non_shaded_mask_file[0]

        # Derive file-specific paths
        model_save_path_index = os.path.join(model_save_dir, f"{base_name}_rf_model_CB.pkl")
        output_masked_snow_map_path = os.path.join(output_dir, f"{base_name}_RF_predicted_snow_map_shaded.tif")

        # Find the corresponding true extent raster
        true_extent_raster_path = os.path.join(true_extent_raster_dir, f"{base_name}_PS_TOAR_8b_gaussian.tif")

        # Check if true_extent_raster_path exists
        if not os.path.exists(true_extent_raster_path):
            print(f"True extent raster not found for {CB_tif_file}, skipping.")
            continue

        # Print the current file being processed for debugging
        print(f"Processing file: {CB_tif_file}")

        # Process the file independently
        process_file(CB_tif_file, non_shaded_mask_file, true_extent_raster_path, output_masked_snow_map_path, model_save_path_index)

if __name__ == "__main__":
    main()
