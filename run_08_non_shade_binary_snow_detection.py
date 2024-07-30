import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from osgeo import gdal
import os
from datetime import datetime

def plot_histogram(data, bins=50, date_str=""):
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    plt.xlabel('Index Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Index - {date_str}')
    plt.show()

def identify_snow_threshold(data, bins=50, smooth_window=11, poly_order=2):
    histogram, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth the histogram
    smooth_histogram = savgol_filter(histogram, smooth_window, poly_order)
    
    # Calculate the first derivative
    first_derivative = np.gradient(smooth_histogram)
    
    # Identify the point where the first derivative has a significant decrease
    threshold_index = np.where(first_derivative < 0)[0][0]
    threshold_value = bin_centers[threshold_index]
    
    return threshold_value

def classify_snow(data, threshold_value):
    snow_mask = data <= threshold_value
    classified_data = np.where(snow_mask, 1, 0)
    return classified_data

def process_no_shade_raster(file_path, snow_output_dir):
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
    
    # Plot the histogram
    plot_histogram(valid_data, date_str=date_str)
    
    # Identify the snow threshold
    threshold_value = identify_snow_threshold(valid_data)
    
    # Print the threshold value found for the current date
    print(f"Date: {date_str}, Threshold: {threshold_value}")
    
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

def process_no_shade_directory(input_directory, snow_output_dir):
    for filename in os.listdir(input_directory):
        if filename.endswith('_without_shadow.tif'):
            file_path = os.path.join(input_directory, filename)
            process_no_shade_raster(file_path, snow_output_dir)

# Path to your directory containing the no-shade .tif files
input_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shade_no_shade_8b_TOAR/non_shaded'

# Define output directory for snow classified images
snow_output_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/snow_classified/non_shaded'

process_no_shade_directory(input_directory, snow_output_directory)
