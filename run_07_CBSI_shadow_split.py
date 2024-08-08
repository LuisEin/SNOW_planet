# -*- coding: utf-8 -*-
'''
Created on Monday Jul 29 10:23:12 2024

This file takes Coastal Blue Snow Index calculated GeoTiff, analyses its 
Histogram and locates the most negative significant peak.
Then splits it off as these are the potentially shaded areas.
Saves the shaded and the non-shaded areas as new files.

@author: luis
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from osgeo import gdal
import os
from datetime import datetime

def plot_histogram(data, bins=50, date_str="", threshold_value=None):
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    if threshold_value is not None:
        plt.axvline(x=threshold_value, color='red', linestyle='--', 
                    label=f'Threshold: {threshold_value:.2f}, \nOffset Value: {offset_value:.2f}')
        plt.legend()
    plt.xlabel('Coastal Blue Index')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Coastal Blue Index - {date_str}')
    plt.show()

def identify_peaks_and_valleys(data, bins=50, height_threshold=0.05, prominence_threshold=0.01, width_threshold=1):
    histogram, bin_edges = np.histogram(data, bins=bins)
    
    # Adjust the peak detection to be more sensitive
    peaks, _ = find_peaks(histogram, height=height_threshold * np.max(histogram), prominence=prominence_threshold, width=width_threshold)
    valleys, _ = find_peaks(-histogram, prominence=prominence_threshold, width=width_threshold)  # Invert histogram to find valleys
    
    return peaks, valleys, bin_edges

def split_shadow_area(data, peaks, valleys, bin_edges, offset=0.0):
    most_negative_peak_index = peaks[0]
    
    # Find the valley between the most negative peak and the next significant peak
    valley_indices = [v for v in valleys if v > most_negative_peak_index]
    if valley_indices:
        valley_index = valley_indices[0]
    else:
        valley_index = len(bin_edges) - 2
    
    threshold_value = bin_edges[valley_index + 1] + offset
    
    shadow_mask = data <= threshold_value
    shadow_area = np.copy(data)
    shadow_area[~shadow_mask] = np.nan
    original_without_shadow = np.copy(data)
    original_without_shadow[shadow_mask] = np.nan
    return shadow_area, original_without_shadow, threshold_value

def process_raster(file_path, shadow_output_dir, non_shadow_output_dir, offset=0.0):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)

    # Extract date from file name or use current date
    file_name = os.path.basename(file_path)
    date_str = datetime.now().strftime('%Y-%m-%d')
    if "_" in file_name:
        date_str = file_name.split("_")[0]
    
    # Flatten the array and remove NaN values for histogram
    valid_data = data[~np.isnan(data)].flatten()

    # Identify peaks and valleys in the histogram
    peaks, valleys, bin_edges = identify_peaks_and_valleys(valid_data)

    # Split the shadow area with the offset
    shadow_area, original_without_shadow, threshold_value = split_shadow_area(data, peaks, valleys, bin_edges, offset=offset)

    # Plot the histogram with the threshold value
    plot_histogram(valid_data, date_str=date_str, threshold_value=threshold_value)

    # Define output file paths
    shadow_output_path = os.path.join(shadow_output_dir, file_name.replace('.tif', '_shadow_area.tif'))
    non_shadow_output_path = os.path.join(non_shadow_output_dir, file_name.replace('.tif', '_without_shadow.tif'))

    # Save the shadow area and the updated original image
    save_raster(shadow_area, shadow_output_path, dataset)
    save_raster(original_without_shadow, non_shadow_output_path, dataset)

def save_raster(data, output_path, reference_dataset):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(
        output_path,
        reference_dataset.RasterXSize,
        reference_dataset.RasterYSize,
        1,
        gdal.GDT_Float32
    )
    out_dataset.SetGeoTransform(reference_dataset.GetGeoTransform())
    out_dataset.SetProjection(reference_dataset.GetProjection())
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(data)
    out_band.SetNoDataValue(np.nan)
    out_dataset.FlushCache()
    out_dataset = None

def process_directory(input_directory, shadow_output_dir, non_shadow_output_dir, offset=0.0):
    for filename in os.listdir(input_directory):
        if filename.endswith('.tif'):
            file_path = os.path.join(input_directory, filename)
            process_raster(file_path, shadow_output_dir, non_shadow_output_dir, offset=offset)

# Paths to your directories
input_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/TOAR_gaussian_filtered/CBSI'
shadow_output_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shade_no_shade_8b_TOAR_gaussian_filtered/shaded'
non_shadow_output_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shade_no_shade_8b_TOAR_gaussian_filtered/non_shaded'

os.makedirs(shadow_output_directory, exist_ok=True)
os.makedirs(non_shadow_output_directory, exist_ok=True)

# Example usage with an offset
offset_value = 0.045  # Adjust this value as needed
process_directory(input_directory, shadow_output_directory, non_shadow_output_directory, offset=offset_value)
