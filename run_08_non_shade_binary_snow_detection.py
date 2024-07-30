import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from osgeo import gdal
import os
from datetime import datetime

def plot_histogram(data, bins=50, date_str="", threshold_value=None):
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    plt.xlabel('Index Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Index - {date_str}')
    if threshold_value is not None:
        plt.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold_value}')
        plt.legend()
    plt.show()

def identify_peak_characteristics(histogram, bin_centers):
    peaks, _ = find_peaks(histogram)
    
    if not peaks.size:
        return None, False, 0
    
    # Find the most negative peak
    most_negative_peak_index = peaks[0]
    for peak in peaks:
        if bin_centers[peak] < bin_centers[most_negative_peak_index]:
            most_negative_peak_index = peak
    
    # Determine peak characteristics
    peak_value = histogram[most_negative_peak_index]
    left_base = np.where(histogram[:most_negative_peak_index] <= peak_value / 2)[0]
    right_base = np.where(histogram[most_negative_peak_index:] <= peak_value / 2)[0]
    
    if left_base.size == 0 or right_base.size == 0:
        peak_width = np.inf
    else:
        peak_width = (right_base[0] + most_negative_peak_index) - left_base[-1]
    
    steep_and_lonely = peak_value > 2 * np.mean(histogram) and peak_width < len(histogram) / 10
    
    return most_negative_peak_index, steep_and_lonely, peak_value

def identify_snow_threshold(data, bins=50, smooth_window=11, poly_order=2, offset_steep=0.5, offset_flat=1.0):
    histogram, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth the histogram
    smooth_histogram = savgol_filter(histogram, smooth_window, poly_order)
    
    # Identify peak characteristics
    most_negative_peak_index, steep_and_lonely, peak_value = identify_peak_characteristics(smooth_histogram, bin_centers)
    
    if most_negative_peak_index is None:
        raise ValueError("No peaks found in the histogram")
    
    # Ensure we move to the descending side of the most negative peak
    first_derivative = np.gradient(smooth_histogram)
    descending_side_index = most_negative_peak_index
    while descending_side_index < len(first_derivative) - 1 and first_derivative[descending_side_index] <= 0:
        descending_side_index += 1
    
    # Determine offset based on peak characteristics
    offset = offset_steep if steep_and_lonely else offset_flat
    threshold_index = descending_side_index + offset
    
    if threshold_index < len(bin_centers) - 1:
        lower_bin_center = bin_centers[int(threshold_index)]
        upper_bin_center = bin_centers[int(threshold_index) + 1]
        threshold_value = lower_bin_center + (upper_bin_center - lower_bin_center) * (threshold_index - int(threshold_index))
    else:
        threshold_value = bin_centers[-1]
    
    # Ensure the threshold is between -0.4 and -0.15
    threshold_value = max(min(threshold_value, -0.15), -0.4)
    
    # If the threshold is out of bounds, assume peak is flat and adjust accordingly
    if not (-0.4 <= threshold_value <= -0.15):
        offset = offset_flat
        threshold_index = descending_side_index + offset
        if threshold_index < len(bin_centers) - 1:
            lower_bin_center = bin_centers[int(threshold_index)]
            upper_bin_center = bin_centers[int(threshold_index) + 1]
            threshold_value = lower_bin_center + (upper_bin_center - lower_bin_center) * (threshold_index - int(threshold_index))
        else:
            threshold_value = bin_centers[-1]
        threshold_value = max(min(threshold_value, -0.15), -0.4)
    
    # Print statements for debugging
    print(f"Threshold index before offset: {descending_side_index}")
    print(f"Threshold index after offset: {threshold_index}")
    print(f"Adjusted threshold value: {threshold_value}")
    
    return threshold_value

def classify_snow(data, threshold_value):
    snow_mask = data <= threshold_value
    classified_data = np.where(snow_mask, 1, 0)
    return classified_data

def process_no_shade_raster(file_path, snow_output_dir, offset_steep=0.5, offset_flat=1.0):
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
    
    # Identify the snow threshold with an offset
    threshold_value = identify_snow_threshold(valid_data, offset_steep=offset_steep, offset_flat=offset_flat)
    
    # Print the date and threshold value found for the current date
    print(f"Date: {date_str}, Threshold: {threshold_value}")
    
    # Plot the histogram with the threshold value
    plot_histogram(valid_data, date_str=date_str, threshold_value=threshold_value)
    
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

def process_no_shade_directory(input_directory, snow_output_dir, offset_steep=0.5, offset_flat=1.0):
    for filename in os.listdir(input_directory):
        if filename.endswith('_without_shadow.tif'):
            file_path = os.path.join(input_directory, filename)
            process_no_shade_raster(file_path, snow_output_dir, offset_steep, offset_flat)

# Path to your directory containing the no-shade .tif files
input_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shade_no_shade_8b_TOAR/non_shaded'

# Define output directory for snow classified images
snow_output_directory = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/snow_classified/non_shaded'

# Define the offset values to adjust the threshold
offset_steep_value = 0.5  # Adjust this value as needed
offset_flat_value = 1.0  # Adjust this value as needed

process_no_shade_directory(input_directory, snow_output_directory, offset_steep=offset_steep_value, offset_flat=offset_flat_value)
