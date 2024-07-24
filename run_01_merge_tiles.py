# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 09 17:13:12 2024

This file takes PlanetScope Scenes by the same date and puts the single tiles 
together.
Saves them as one tif file containing the whole AOI.

This is the first script to use when starting an analysis of PlanetScope data

@author: luis
'''

import os
import glob
import numpy as np
from osgeo import gdal
import shutil
from functions_planet import read_shapefile, extract_time_from_filename, check_coverage

# Define paths
# 8 Band: *AnalyticMS_SR_8b_clip.tif
# Läuft irgendwie im Moment nur gut mit 8 Band Daten
input_pattern = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_raw/4_band/Data_March-June_23_Feb-March_24_psscene_analytic_sr_udm2/PSScene/*AnalyticMS_SR_clip.tif'
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/4_band/Data_March-June_23_Feb-March_24_psscene_analytic_sr_udm2'
shapefile_path = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Shapefiles/shapefile_Zugspitze/03_AOI_shp_zugspitze_reproj_for_code/AOI_zugspitze_reproj_32632.shp'  

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
        output_path = os.path.join(output_dir, f"{date_str}_{time_str}_PS_ready.tif")
        shutil.copyfile(largest_file, output_path)
        print(f"Single image {largest_file} covers the AOI. Copied to {output_path}")
        continue

    # Otherwise, clip the images to fill the AOI
    for i, file in enumerate(file_list):
        output_path = os.path.join(output_dir, f"{date_str}_clip_{i}.tif")
        gdal.Warp(output_path, file, cutlineDSName=shapefile_path, cropToCutline=True, dstNodata=np.nan)
        print(f"Clipped image {file} to AOI and saved to {output_path}")

print("Processing complete.")
