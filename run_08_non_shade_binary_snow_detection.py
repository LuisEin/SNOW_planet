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
Identify_snow_threshold() Function, to detect a significant change in the pattern.
Empirically set to 0.1

@author: luis 
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from osgeo import gdal
import os
from datetime import datetime

def plot_histogram(data, bins=50, date_str="", threshold_value=None, percentage_threshold_value=None, bin_shift_value=None):
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    if threshold_value is not None:
        plt.axvline(x=threshold_value, color='red', linestyle='--', label=f'Threshold: {threshold_value:.2f}')
    plt.xlabel('Index Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Coastal Blue Index - {date_str}')
    if threshold_value is not None:
        plt.text(0.95, 0.95, f'Threshold: {threshold_value:.2f}\nPercentage Threshold: {percentage_threshold_value}\nBin Shift: {bin_shift_value}',
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
    plt.legend()
    plt.show()

def identify_snow_threshold(data, bins=50, percentage_threshold=0.1, bin_shift_value=1):
    histogram, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Identify all peaks with adjusted sensitivity
    peaks, _ = find_peaks(histogram, height=0, threshold=0, distance=1, prominence=1e-6, width=1)
    
    # Find the most negative peak (leftmost peak)
    most_negative_peak_index = peaks[0]
    for peak in peaks:
        if bin_centers[peak] < bin_centers[most_negative_peak_index]:
            most_negative_peak_index = peak
    
    # Find the highest bin within the leftmost peak
    highest_bin_index = most_negative_peak_index
    
    # Calculate the percentage decrease from the peak to the next bins
    threshold_index = highest_bin_index
    previous_decreases = []

    while threshold_index < len(histogram) - 1:
        current_height = histogram[threshold_index]
        next_height = histogram[threshold_index + 1]
        percentage_decrease = (current_height - next_height) / current_height * 100
        
        # Special condition: Skip storing and proceed if decrease is smaller than 5%
        if threshold_index == highest_bin_index and percentage_decrease < 5:
            threshold_index += 1
            continue
        
        # Store the percentage decrease
        previous_decreases.append(percentage_decrease)

        # Log information for debugging
        print(f"Highest Bin: {highest_bin_index}, Bin: {threshold_index}, Current Height: {current_height}, Next Height: {next_height}, Percentage Decrease: {percentage_decrease:.2f}%")
        
        # Check if the current percentage decrease is significantly smaller than the average of the previous decreases
        if len(previous_decreases) > 1:
            avg_previous_decreases = np.mean(previous_decreases[:-1])
            if percentage_decrease <= avg_previous_decreases * (1 - percentage_threshold):
                break
        
        threshold_index += 1
    
    # Calculate the quadratic shift based on the distance from the peak bin
    distance_from_peak = threshold_index - highest_bin_index
    if distance_from_peak > 0:
        bin_shift = int(bin_shift_value / (distance_from_peak ** 0.5))
    else:
        bin_shift = 0
    threshold_index = min(threshold_index + bin_shift, len(histogram) - 1)
    
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

def process_no_shade_raster(file_path, snow_output_dir, percentage_threshold=0.1, bin_shift_value=1):
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
    threshold_value = identify_snow_threshold(valid_data, percentage_threshold=percentage_threshold, bin_shift_value=bin_shift_value)
    
    # Plot the histogram with the threshold value
    plot_histogram(valid_data, date_str=date_str, threshold_value=threshold_value, percentage_threshold_value=percentage_threshold, bin_shift_value=bin_shift_value)
    
    # Print the date and threshold value found for the current date
    print(f"Date: {date_str}\n----------------\n\n\n") # , Threshold: {threshold_value}
    
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

def process_no_shade_directory(input_directory, snow_output_dir, percentage_threshold=0.1, bin_shift_value=1):
    for filename in os.listdir(input_directory):
        if filename.endswith('_without_shadow.tif'):
            file_path = os.path.join(input_directory, filename)
            process_no_shade_raster(file_path, snow_output_dir, percentage_threshold=percentage_threshold, bin_shift_value=bin_shift_value)

# Path to your directory containing the no-shade .tif files
input_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shade_no_shade_8b_TOAR_gaussian_filtered/non_shaded_offset_0.02'

# Define output directory for snow classified images
snow_output_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/snow_classified_gaussian_filtered/non_shaded_offset_0.02'

os.makedirs(input_directory, exist_ok=True)
os.makedirs(snow_output_directory, exist_ok=True)

# Define the percentage threshold value to adjust the threshold
percentage_threshold_value = 0.1  # Adjust this value as needed

# Define the bin shift value to adjust the threshold
bin_shift_value = 3  # Adjust this value as needed

process_no_shade_directory(input_directory, snow_output_directory, percentage_threshold=percentage_threshold_value, bin_shift_value=bin_shift_value)
