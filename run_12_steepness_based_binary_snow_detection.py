# -*- coding: utf-8 -*-
"""
This script processes PlanetScope data to classify snow-covered regions based 
on steepness classes. The steps are as follows:

1. Load a steepness mask and resample it to match the dimensions of the input 
   GeoTiff file.
2. Clip the input raster into three steepness classes: 0°-30°, 31°-50°, and 51°-90°.
3. For each class:
   a. Generate a histogram of pixel intensities.
   b. Attempt to identify a threshold for snow detection by finding peaks and 
      valleys in the histogram.
   c. Save the histogram to disk, regardless of whether a threshold was found.
   d. Classify pixels as snow-covered (1) or not (0) based on the threshold.
   e. Save the binary classification as a temporary GeoTiff file.
4. Combine the three binary classifications into a single output GeoTiff.
5. Process all GeoTiff files in the input directory and save results in the 
   output directory.

Parameters:
- input_directory: Directory with input GeoTiff files.
- output_directory: Directory to save combined classified GeoTiff files.
- steepness_mask_path: Path to the GeoTiff file containing steepness classes.
- height_threshold: Minimum peak height for histogram peak detection.
- histogram_output_base_dir: Directory to save histograms and intermediate results.

Usage: Adjust paths and height_threshold, then run the script.

@author: Luis
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from osgeo import gdal, gdal_array
import os
from datetime import datetime

def plot_histogram_with_threshold(data, bins=50, threshold=None, date_str="", output_path=None, steepness_label=None):
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
    plt.xlabel('Coastal Blue Intensity')
    plt.ylabel('Frequency')
    title = f'Histogram of Coastal Blue Band in Shaded Areas \n {date_str}'
    if steepness_label:
        title += f' (Steepness-class: {steepness_label[0]}°-{steepness_label[1]}°)'
    plt.title(title)
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()

def identify_threshold(data, bins=50, prominence_threshold=0.01, width_threshold=1, height_threshold=None):
    histogram, bin_edges = np.histogram(data, bins=bins)
    
    # Exclude boundary values between 0 and 6000 with low frequencies
    mask = (bin_edges[:-1] >= 6000) | (np.diff(bin_edges) > 100)
    histogram = histogram[mask]
    bin_edges = bin_edges[:-1][mask]
    
    # Identify peaks
    peaks, _ = find_peaks(histogram, prominence=prominence_threshold, width=width_threshold, height=height_threshold)
    valleys, _ = find_peaks(-histogram, prominence=prominence_threshold, width=width_threshold)  # Invert histogram to find valleys
    
    # Choose the valley between the two peaks as the threshold
    if len(peaks) >= 2:
        left_peak = peaks[0]
        right_peak = peaks[1]
        
        valley_indices = [v for v in valleys if left_peak < v < right_peak]
        if valley_indices:
            valley_index = valley_indices[0]
        else:
            valley_index = (left_peak + right_peak) // 2
        
        threshold_value = bin_edges[valley_index]
    else:
        raise ValueError("Unable to identify two distinct peaks in the histogram")
    
    return threshold_value

def binary_snow_classification(data, threshold):
    binary_data = np.full(data.shape, np.nan, dtype=np.float32)
    binary_data[data <= threshold] = 0
    binary_data[data > threshold] = 1
    return binary_data

def save_raster(data, output_path, reference_dataset):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(
        output_path,
        reference_dataset.RasterXSize,
        reference_dataset.RasterYSize,
        1,
        gdal.GDT_Float32  # Save as float data type to accommodate NaN values
    )
    out_dataset.SetGeoTransform(reference_dataset.GetGeoTransform())
    out_dataset.SetProjection(reference_dataset.GetProjection())
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(data)
    out_band.SetNoDataValue(np.nan)
    out_dataset.FlushCache()
    out_dataset = None

def resample_steepness_mask(steepness_mask_path, target_dataset):
    steepness_dataset = gdal.Open(steepness_mask_path)
    target_projection = target_dataset.GetProjection()
    target_geotransform = target_dataset.GetGeoTransform()
    target_x_size = target_dataset.RasterXSize
    target_y_size = target_dataset.RasterYSize

    resampled_steepness_path = '/vsimem/resampled_steepness.tif'
    gdal.Warp(
        resampled_steepness_path,
        steepness_dataset,
        format='GTiff',
        width=target_x_size,
        height=target_y_size,
        dstSRS=target_projection,
        dstNodata=np.nan,
        xRes=target_geotransform[1],
        yRes=-target_geotransform[5],
        outputBounds=(
            target_geotransform[0],
            target_geotransform[3] + target_y_size * target_geotransform[5],
            target_geotransform[0] + target_x_size * target_geotransform[1],
            target_geotransform[3]
        )
    )
    
    resampled_steepness_dataset = gdal.Open(resampled_steepness_path)
    resampled_steepness_mask = resampled_steepness_dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
    return resampled_steepness_mask

def process_raster(file_path, output_dir, height_threshold, steepness_mask_path, steepness_value, histogram_output_dir, without_threshold_dir, steepness_label, temp_output_dir):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)

    # Resample steepness mask to match the input raster file dimensions
    steepness_mask = resample_steepness_mask(steepness_mask_path, dataset)

    # Apply steepness mask
    masked_data = np.where(steepness_mask == steepness_value, data, np.nan)

    # Exclude boundary values between 0 and 6000 with low frequencies
    valid_data = masked_data[(masked_data >= 6000) | np.isnan(masked_data)]
    
    # Save temporary masked data
    temp_file_name = f'{os.path.splitext(os.path.basename(file_path))[0]}_CB_{steepness_label[0]}_{steepness_label[1]}.tif'
    temp_output_path = os.path.join(temp_output_dir, temp_file_name)
    save_raster(masked_data, temp_output_path, dataset)

    # Extract date from file name or use current date
    file_name = os.path.basename(file_path)
    date_str = datetime.now().strftime('%Y-%m-%d')
    if "_" in file_name:
        date_str = file_name.split("_")[0]
    
    # Flatten the array and remove NaN values for histogram
    valid_data = valid_data[~np.isnan(valid_data)].flatten()

    # Identify threshold and save histogram regardless of success
    histogram_file_name = f'{date_str}_{file_name}_CB_{steepness_label[0]}_{steepness_label[1]}_histogram.png'
    histogram_output_path = os.path.join(histogram_output_dir, histogram_file_name)
    without_threshold_output_path = os.path.join(without_threshold_dir, histogram_file_name)
    threshold_value = None

    try:
        threshold_value = identify_threshold(valid_data, height_threshold=height_threshold)
    except ValueError as e:
        print(f"Error processing file {file_path}: {e}")
    finally:
        # Save histogram to both directories based on threshold detection
        plot_histogram_with_threshold(valid_data, threshold=threshold_value, date_str=date_str, output_path=histogram_output_path, steepness_label=steepness_label)
        if threshold_value is None:
            plot_histogram_with_threshold(valid_data, threshold=None, date_str=date_str, output_path=without_threshold_output_path, steepness_label=steepness_label)

    if threshold_value is None:
        return None  # If no threshold was found, return None

    # Perform binary snow classification
    binary_data = binary_snow_classification(masked_data, threshold_value)

    return binary_data

def combine_rasters(raster_list, output_path, reference_dataset):
    combined_data = np.nanmax(raster_list, axis=0)  # Combine using max value where data overlaps
    save_raster(combined_data, output_path, reference_dataset)

def process_directory(input_directory, output_directory, height_threshold, steepness_mask_path, histogram_output_base_dir):
    steepness_classes = [(0, 30), (31, 50), (51, 90)]  # Adjusted to cover entire range without gaps

    for filename in os.listdir(input_directory):
        if filename.endswith('.tif'):
            file_path = os.path.join(input_directory, filename)
            try:
                combined_rasters = []
                dataset = gdal.Open(file_path)
                reference_dataset = dataset  # Keep reference for saving combined output

                for steepness_value, steepness_label in enumerate(steepness_classes, start=1):
                    steepness_dir = os.path.join(output_directory, f'steepness_{steepness_label[0]}_{steepness_label[1]}')
                    os.makedirs(steepness_dir, exist_ok=True)

                    histogram_output_dir = os.path.join(histogram_output_base_dir, f'steepness_{steepness_label[0]}_{steepness_label[1]}')
                    os.makedirs(histogram_output_dir, exist_ok=True)

                    without_threshold_dir = os.path.join(histogram_output_base_dir, f'steepness_{steepness_label[0]}_{steepness_label[1]}_without_threshold')
                    os.makedirs(without_threshold_dir, exist_ok=True)

                    temp_output_dir = os.path.join(histogram_output_base_dir, f'steepness_{steepness_label[0]}_{steepness_label[1]}_temp')
                    os.makedirs(temp_output_dir, exist_ok=True)
                    
                    binary_data = process_raster(file_path, steepness_dir, height_threshold, steepness_mask_path, steepness_value, histogram_output_dir, without_threshold_dir, steepness_label, temp_output_dir)

                    if binary_data is not None:
                        combined_rasters.append(binary_data)

                # Combine all steepness classes into one output
                if combined_rasters:
                    output_file_name = f'{os.path.splitext(os.path.basename(file_path))[0]}_CB_combined_snow_classified.tif'
                    output_path = os.path.join(output_directory, output_file_name)
                    combine_rasters(combined_rasters, output_path, reference_dataset)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Paths to your directories
input_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shadow_masked_original_ready_files_gaussian_filtered_offset_0.02' #_gaussian_filtered
output_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/snow_classified/steepness_dependend/shaded_less_shaded'
steepness_mask_path = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/DTM/steepness_classes_30_50_90.tif'
histogram_output_base_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/code/temp'

os.makedirs(output_directory, exist_ok=True)

# Set the height threshold for peak detection
height_threshold = 200  # Adjust this value as needed

process_directory(input_directory, output_directory, height_threshold, steepness_mask_path, histogram_output_base_dir)
