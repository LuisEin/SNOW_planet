# -*- coding: utf-8 -*-
'''
Created on Wednesday Jul 31 10:23:12 2024

This file takes non_shaded CBSI data that has been processed in run_07_shadow_split.
It creates a Histogram and looks for the optimal Threshold of where the 
Peak with the snowy values ends.
This works with a loop, that iterates over each bin right of the peak
And calculates the percentage decrease each time, stores it in a list.
If the current percentage decrease is significantly smaller than the average 
In the list, the loop is stopped and threshold is set. 
The significance is calculated with the percentage threshold in the
Identify_snow_threshold() Function, to detedct a significant change in the pattern.
Empirically set to 0.1

@author: luis 
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from osgeo import gdal
import os
from datetime import datetime

def plot_histogram(data, bins=50, date_str="", threshold_value=None, offset=None):
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    if threshold_value is not None:
        plt.axvline(x=threshold_value, color='red', linestyle='--', label=f'Threshold: {threshold_value:.2f}')
    plt.xlabel('Index Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Coastal Blue Index - {date_str}')
    if threshold_value is not None:
        plt.text(0.95, 0.95, f'Threshold: {threshold_value:.2f}\nOffset: {offset}',
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
    plt.legend()
    plt.show()

def identify_snow_threshold(data, bins=50, smooth_window=11, poly_order=2, percentage_threshold=0.5, min_bins_from_peak=3):
    histogram, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth the histogram
    smooth_histogram = savgol_filter(histogram, smooth_window, poly_order)
    
    # Identify all peaks
    peaks, _ = find_peaks(smooth_histogram)
    
    # Find the most negative peak
    most_negative_peak_index = peaks[0]
    for peak in peaks:
        if bin_centers[peak] < bin_centers[most_negative_peak_index]:
            most_negative_peak_index = peak

    # Calculate the percentage decrease from the peak to the next bins
    threshold_index = most_negative_peak_index + min_bins_from_peak
    previous_decreases = []

    while threshold_index < len(smooth_histogram) - 1:
        current_height = smooth_histogram[threshold_index]
        next_height = smooth_histogram[threshold_index + 1]
        percentage_decrease = (current_height - next_height) / current_height * 100
        
        # Store the percentage decrease
        previous_decreases.append(percentage_decrease)

        # Check if the current percentage decrease is significantly smaller than the average of the previous decreases
        if len(previous_decreases) > 1:
            avg_previous_decreases = np.mean(previous_decreases[:-1])
            if percentage_decrease <= avg_previous_decreases * (1 - percentage_threshold):
                break
        
        threshold_index += 1
    
    threshold_value = bin_centers[threshold_index]
    
    # Print statements for debugging
    print(f"Threshold index: {threshold_index}")
    print(f"Threshold value: {threshold_value}")
    
    return threshold_value

def classify_snow(data, threshold_value):
    classified_data = np.full(data.shape, np.nan, dtype=np.float32)
    classified_data[data <= threshold_value] = 1
    classified_data[data > threshold_value] = 0
    return classified_data

def process_no_shade_raster(file_path, snow_output_dir, percentage_threshold=0.5, min_bins_from_peak=3):
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
    
    # Identify the snow threshold with the given percentage threshold
    threshold_value = identify_snow_threshold(valid_data, percentage_threshold=percentage_threshold, min_bins_from_peak=min_bins_from_peak)
    
    # Plot the histogram with the threshold value
    plot_histogram(valid_data, date_str=date_str, threshold_value=threshold_value)
    
    # Print the date and threshold value found for the current date
    print(f"Date: {date_str}") # , Threshold: {threshold_value}
    
    # Classify the data
    classified_data = classify_snow(data, threshold_value)
    
    # Define output file path
    snow_output_path = os.path.join(snow_output_dir, file_name.replace('.tif', '_snow_classified.tif'))
    
    # Save the classified image
    save_raster(classified_data, snow_output_path, dataset)

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

def process_no_shade_directory(input_directory, snow_output_dir, percentage_threshold=0.5, min_bins_from_peak=3):
    for filename in os.listdir(input_directory):
        if filename.endswith('_without_shadow.tif'):
            file_path = os.path.join(input_directory, filename)
            process_no_shade_raster(file_path, snow_output_dir, percentage_threshold=percentage_threshold, min_bins_from_peak=min_bins_from_peak)

# Path to your directory containing the no-shade .tif files
input_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shade_no_shade_8b_TOAR/non_shaded'

# Define output directory for snow classified images
snow_output_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/snow_classified/non_shaded'

# Define the percentage threshold value to adjust the threshold
percentage_threshold_value = 0.1  # Adjust this value as needed

# Define the minimum number of bins from the peak
min_bins_from_peak_value = 1  # Adjust this value as needed

process_no_shade_directory(input_directory, snow_output_directory, percentage_threshold=percentage_threshold_value, min_bins_from_peak=min_bins_from_peak_value)
