import os
from osgeo import gdal
import numpy as np
from datetime import datetime, timedelta

# Paths
sws_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Sentinel_Data/SWS/SWS_all_data_processed/binary_1wet_0dry'
non_shaded_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/binary_classification/non_shaded'
shaded_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/binary_classification/shaded'
combined_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Planet_Sentinel_combined/same_date'
output_dir_1day = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Planet_Sentinel_combined/+-1_day'
rf_merged_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Planet_Sentinel_combined/predicted_merged_RF_binary_snow_maps'

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

# Function to extract date and time from SWS filenames
def extract_date_time_from_sws_filename(file_name):
    parts = file_name.split('_')
    date = f"{parts[1]}-{parts[2]}-{parts[3]}"
    time = f"{parts[4]}h_{parts[5]}"
    return date, time

# Function to extract date from PS filenames
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

# Function to find SWS file ±1 day from a given date
def find_sws_file_within_one_day(ps_date_str, sws_dir):
    ps_date = datetime.strptime(ps_date_str, "%Y-%m-%d")
    candidates = []
    for file_name in os.listdir(sws_dir):
        sws_date_str, sws_time = extract_date_time_from_sws_filename(file_name)
        sws_date = datetime.strptime(sws_date_str, "%Y-%m-%d")
        if abs((ps_date - sws_date).days) <= 1:
            candidates.append((sws_date_str, sws_time, os.path.join(sws_dir, file_name)))
    return candidates

# Function to process and combine the data
def process_files(date, non_shaded_file, shaded_file, sws_file, sws_time, output_path, is_1day=False):
    try:
        # Combine non-shaded and shaded files
        non_shaded_arr, geo_transform, projection = read_tif(non_shaded_file)
        shaded_arr, _, _ = read_tif(shaded_file)

        # Ensure that the arrays have the same shape
        if non_shaded_arr.shape != shaded_arr.shape:
            min_rows = min(non_shaded_arr.shape[0], shaded_arr.shape[0])
            min_cols = min(non_shaded_arr.shape[1], shaded_arr.shape[1])
            non_shaded_arr = non_shaded_arr[:min_rows, :min_cols]
            shaded_arr = shaded_arr[:min_rows, :min_cols]

        # Combine the arrays by keeping valid data and avoiding NoData values (255)
        combined_arr = np.where(non_shaded_arr == 255, shaded_arr, non_shaded_arr)
        combined_arr = np.where(shaded_arr == 255, non_shaded_arr, combined_arr)

        # Save the combined RF predicted binary snow map
        combined_rf_file_path = os.path.join(rf_merged_dir, f'{date.replace("-", "")}_combined_RF_predicted_binary_snow.tif')
        write_tif(combined_rf_file_path, combined_arr, geo_transform, projection)

        # Look for matching SWS file
        if sws_file:
            print(f"Matching SWS file found for date: {date}")
            
            sws_arr, _, _ = read_tif(sws_file)
            sws_resampled_arr = np.kron(sws_arr, np.ones((20, 20)))  # Resample SWS 60m to match PlanetScope 3m resolution
            
            # Ensure the resampled array has the same shape as the combined array
            if sws_resampled_arr.shape != combined_arr.shape:
                min_rows = min(combined_arr.shape[0], sws_resampled_arr.shape[0])
                min_cols = min(combined_arr.shape[1], sws_resampled_arr.shape[1])
                combined_arr = combined_arr[:min_rows, :min_cols]
                sws_resampled_arr = sws_resampled_arr[:min_rows, :min_cols]

            # Create the combined classification array
            final_arr = np.full(combined_arr.shape, 255, dtype=np.uint8)  # Default to 255 for NoData
            final_arr[(combined_arr == 1) & (sws_resampled_arr == 0)] = 30  # snow and dry
            final_arr[(combined_arr == 1) & (sws_resampled_arr == 1)] = 40  # snow and wet
            final_arr[(combined_arr == 0) & (sws_resampled_arr == 0)] = 60  # nosnow and dry
            final_arr[(combined_arr == 0) & (sws_resampled_arr == 1)] = 80  # nosnow and wet
            
            # Save the final combined array
            if is_1day:
                final_output_path = os.path.join(output_path, f'{date.replace("-", "")}_PS_RF_predicted_{sws_date.replace("-", "")}_{sws_time}_SWS_combined.tif')
            else:
                final_output_path = os.path.join(output_path, f'{date.replace("-", "")}_PS_RF_predicted_{sws_date.replace("-", "")}_{sws_time}_SWS_combined.tif')
            write_tif(final_output_path, final_arr, geo_transform, projection)
        else:
            print(f"No matching SWS file found for date: {date}")
    
    except Exception as e:
        print(f"Error processing files for date {date}: {e}")

# Loop through the non-shaded directory
for non_shaded_file in sorted(os.listdir(non_shaded_dir)):
    if not non_shaded_file.endswith('.tif') or non_shaded_file.endswith('.aux.xml'):
        continue  # Skip non-TIF files or auxiliary files

    date = extract_date_from_filename(non_shaded_file, 'non_shaded')
    
    # Find the corresponding shaded file
    shaded_file = find_matching_shaded_file(date, shaded_dir)
    
    if shaded_file and shaded_file.endswith('.tif') and not shaded_file.endswith('.aux.xml'):
        non_shaded_file_path = os.path.join(non_shaded_dir, non_shaded_file)

        # Process matching date
        matching_sws_file = find_matching_sws_file(date, sws_dir)
        if matching_sws_file:
            sws_date, sws_time = extract_date_time_from_sws_filename(os.path.basename(matching_sws_file))
            process_files(date, non_shaded_file_path, shaded_file, matching_sws_file, sws_time, combined_dir)
        
        # Process ±1 day matches
        sws_candidates = find_sws_file_within_one_day(date, sws_dir)
        for sws_date, sws_time, sws_file_path in sws_candidates:
            process_files(date, non_shaded_file_path, shaded_file, sws_file_path, sws_time, output_dir_1day, is_1day=True)
    else:
        print(f"No matching shaded file found for date: {date}. Skipping to the next date.")
