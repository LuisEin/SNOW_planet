# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 09 17:13:12 2024

This file takes PlanetScope Scenes by the same date, sorts out for the largest
Which covers the biggest area.
Then applies water mask of Eibsee and Frillensee.
Checks if AOI is covered.
Saves them as a tif file.

This is the first script to use when starting an analysis of PlanetScope data

@author: luis
'''

import glob, shutil, re, os
import numpy as np
from osgeo import gdal, ogr
from functions_planet import read_shapefile, apply_water_mask, extract_time_from_filename, check_coverage


# Define paths
# input pattern different for
# SR : *AnalyticMS_SR_8b_clip.tif
# TOAR : *AnalyticMS_8b_clip.tif
# Läuft irgendwie im Moment nur gut mit 8 Band Daten
input_pattern = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_raw/8_band/matching_to_SWS_02_24_until_06-26_TOAR_psscene_analytic_8b_udm2/PSScene/*AnalyticMS_8b_clip.tif'
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band/TOAR'
shapefile_path = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Shapefiles/shapefile_Zugspitze/03_AOI_shp_zugspitze_reproj_for_code/AOI_zugspitze_reproj_32632.shp'
water_mask_path = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Shapefiles/Eibsee/Grid_files/Eibsee_Frillensee_water_mask.tif'
# Define Data Types for naming
band_count = "8b" # 8b or 4b
processing_type = "TOAR" # SR or TOAR

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
# Collect files matching the pattern
files = glob.glob(input_pattern)

# Group files by date
files_by_date = {}
for file in files:
    basename = os.path.basename(file)
    date_str = basename[:8]
    if date_str not in files_by_date:
        files_by_date[date_str] = []
    files_by_date[date_str].append(file)


try:
    aoi_geom = read_shapefile(shapefile_path)
    print(f"AOI Geometry: {aoi_geom.ExportToWkt()}")
except ValueError as e:
    print(e)
    raise

# Read the water mask
water_mask_ds = gdal.Open(water_mask_path)
water_mask_band = water_mask_ds.GetRasterBand(1)
water_mask_array = water_mask_band.ReadAsArray()
water_mask_transform = water_mask_ds.GetGeoTransform()


# Process each date group
for date_str, file_list in files_by_date.items():
    print(f"Processing date: {date_str}")

    # Sort files by size (largest first)
    file_list.sort(key=os.path.getsize, reverse=True)

    # Check if the largest image covers the AOI
    largest_file = file_list[0]
    if check_coverage(largest_file, aoi_geom):
        # The largest image covers the AOI, copy it to output with time in the filename
        time_str = extract_time_from_filename(largest_file)
        output_path = os.path.join(output_dir, f"{date_str}_{time_str}_PS_{processing_type}_{band_count}_ready.tif")
        shutil.copyfile(largest_file, output_path)
        apply_water_mask(output_path, water_mask_array, water_mask_transform)
        print(f"Single image {largest_file} covers the AOI. Copied to {output_path}")
        continue

    # Otherwise, clip the images to fill the AOI
    for i, file in enumerate(file_list):
        output_path = os.path.join(output_dir, f"{date_str}_clip_{i}.tif")
        gdal.Warp(output_path, file, cutlineDSName=shapefile_path, cropToCutline=True, dstNodata=np.nan)
        apply_water_mask(output_path, water_mask_array, water_mask_transform)
        print(f"Clipped image {file} to AOI and saved to {output_path}")

print("Processing complete.")
