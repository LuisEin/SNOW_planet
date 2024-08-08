import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from osgeo import gdal
import os
from datetime import datetime

def plot_histogram_with_threshold(data, bins=50, threshold=None, date_str=""):
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
    plt.xlabel('Coastal Blue Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Coastal Blue Band in Shade - {date_str}')
    plt.legend()
    plt.show()

def identify_threshold(data, bins=50, prominence_threshold=0.01, width_threshold=1, height_threshold=None):
    histogram, bin_edges = np.histogram(data, bins=bins)
    
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

def process_raster(file_path, output_dir, height_threshold):
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

    # Identify threshold
    threshold_value = identify_threshold(valid_data, height_threshold=height_threshold)

    # Plot the histogram with threshold
    plot_histogram_with_threshold(valid_data, threshold=threshold_value, date_str=date_str)

    # Perform binary snow classification
    binary_data = binary_snow_classification(data, threshold_value)

    # Define output file path
    output_path = os.path.join(output_dir, file_name.replace('.tif', '_shaded_parts_snow_classified.tif'))

    # Save the binary classified image
    save_raster(binary_data, output_path, dataset)

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

def process_directory(input_directory, output_directory, height_threshold):
    for filename in os.listdir(input_directory):
        if filename.endswith('.tif'):
            file_path = os.path.join(input_directory, filename)
            try:
                process_raster(file_path, output_directory, height_threshold)
            except ValueError as e:
                print(f"Error processing file {file_path}: {e}")

# Paths to your directories
input_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shadow_masked_original_ready_files_gaussian_filtered'
output_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/snow_classified_gaussian_filtered/shaded'

os.makedirs(output_directory, exist_ok=True)

# Set the height threshold for peak detection
height_threshold = 200  # Adjust this value as needed

process_directory(input_directory, output_directory, height_threshold)
