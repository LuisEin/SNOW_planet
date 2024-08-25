import os
from osgeo import gdal
import numpy as np

# Paths
sws_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Sentinel_Data/SWS/SWS_all_data_processed/binary_1wet_0dry'
non_shaded_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/binary_classification/non_shaded'
shaded_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/binary_classification/shaded'
combined_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/binary_classification/combined'
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Planet_Sentinel_combined'

# Function to read a tif file
def read_tif(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ds = gdal.Open(file_path)
    if ds is None:
        raise ValueError(f"GDAL could not open the file: {file_path}")
    
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    return arr, geo_transform, projection

# Function to write a tif file
def write_tif(file_path, arr, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = arr.shape
    out_ds = driver.Create(file_path, cols, rows, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(arr)
    out_band.FlushCache()

# Function to find matching SWS file
def find_matching_sws_file(date, sws_dir):
    for file_name in os.listdir(sws_dir):
        if date.replace("-", "_") in file_name:
            return os.path.join(sws_dir, file_name)
    return None

# Function to extract date from filenames
def extract_date_from_filename(file_name, file_type):
    if file_type == 'sws':
        return file_name.split('_')[1] + '-' + file_name.split('_')[2] + '-' + file_name.split('_')[3]
    elif file_type in ['non_shaded', 'shaded']:
        return file_name[:4] + '-' + file_name[4:6] + '-' + file_name[6:8]
    return None

# Function to find the corresponding shaded file for a non-shaded file
def find_matching_shaded_file(date, shaded_dir):
    for file_name in os.listdir(shaded_dir):
        if file_name.startswith(date.replace("-", "")):
            return os.path.join(shaded_dir, file_name)
    return None

# Function to process and combine the data
def process_files(date, non_shaded_file, shaded_file):
    try:
        # Combine non-shaded and shaded files
        non_shaded_arr, geo_transform, projection = read_tif(non_shaded_file)
        shaded_arr, _, _ = read_tif(shaded_file)
        
        combined_arr = np.maximum(non_shaded_arr, shaded_arr)  # Combining the arrays
        combined_file_path = os.path.join(combined_dir, f'{date.replace("-", "")}_combined_RF_predicted_binary_snow.tif')
        write_tif(combined_file_path, combined_arr, geo_transform, projection)

        # Look for matching SWS file
        matching_sws_file = find_matching_sws_file(date, sws_dir)
        
        if matching_sws_file:
            print(f"Matching SWS file found for date: {date}")
            
            sws_arr, _, _ = read_tif(matching_sws_file)
            sws_resampled_arr = np.kron(sws_arr, np.ones((20, 20)))  # Resample SWS 60m to match PlanetScope 3m resolution
            
            # Create the combined classification array
            final_arr = np.full(combined_arr.shape, 255, dtype=np.uint8)  # Default to 255 for NoData
            final_arr[(combined_arr == 1) & (sws_resampled_arr == 0)] = 30  # snow and dry
            final_arr[(combined_arr == 1) & (sws_resampled_arr == 1)] = 40  # snow and wet
            final_arr[(combined_arr == 0) & (sws_resampled_arr == 0)] = 60  # nosnow and dry
            final_arr[(combined_arr == 0) & (sws_resampled_arr == 1)] = 80  # nosnow and wet
            
            # Save the final combined array
            final_output_path = os.path.join(output_dir, f'{date.replace("-", "")}_PS_SWS_combined_same_date.tif')
            write_tif(final_output_path, final_arr, geo_transform, projection)
        else:
            print(f"No matching SWS file found for date: {date}")
    
    except Exception as e:
        print(f"Error processing files for date {date}: {e}")

# Loop through the non-shaded directory
for non_shaded_file in sorted(os.listdir(non_shaded_dir)):
    if not non_shaded_file.endswith('.tif'):
        continue  # Skip non-TIF files or auxiliary files

    date = extract_date_from_filename(non_shaded_file, 'non_shaded')
    
    # Find the corresponding shaded file
    shaded_file = find_matching_shaded_file(date, shaded_dir)
    
    if shaded_file:
        non_shaded_file_path = os.path.join(non_shaded_dir, non_shaded_file)
        process_files(date, non_shaded_file_path, shaded_file)
    else:
        print(f"No matching shaded file found for date: {date}. Skipping to the next date.")
