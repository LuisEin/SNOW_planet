# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 09 17:13:12 2024

This script processes PlanetScope Scenes by date and applies Gaussian filtering.
It then converts the images to grayscale. The processed images are saved in
separate directories for filtered and grayscale versions. 

This script is the first step in analyzing PlanetScope data.

Steps:
1. Load the pre-processed PlanetScope images.
2. Apply Gaussian filtering to smooth the images.
3. Convert the images to grayscale.
4. Save the processed images in specified directories.

Note: The input files are assumed to be already clipped and water masked.

Here it still needs to be clarfied on which bands to work - all the bands 

@author: Luis
'''

import glob
import shutil
import re
import os
import numpy as np
import cv2
from osgeo import gdal, ogr
from functions_planet import extract_time_from_filename, apply_gaussian_filter, convert_to_grayscale

# Define paths
input_pattern = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band/TOAR/*_ready.tif'
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band_gaussian_filtered'
output_gray_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band_gaussian_filtered_gray_scaled'
# Define Data Types for naming
band_count = "8b" # 8b or 4b
processing_type = "TOAR" # SR or TOAR

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_gray_dir, exist_ok=True)
# Collect files matching the pattern
files = glob.glob(input_pattern)

# Group files by date
files_by_date = {}
for file in files:
    basename = os.path.basename(file)
    date_str = basename[:8]
    files_by_date[date_str] = file

# Process each date group
for date_str, file in files_by_date.items():
    print(f"Processing date: {date_str}")

    # Copy the image to output with time in the filename
    time_str = extract_time_from_filename(file)
    output_path = os.path.join(output_dir, f"{date_str}_{time_str}_PS_{processing_type}_{band_count}_ready.tif")
    shutil.copyfile(file, output_path)
    
    # Apply Gaussian filter and save
    gaussian_filtered_path = os.path.join(output_dir, f"{date_str}_{time_str}_PS_{processing_type}_{band_count}_gaussian.tif")
    apply_gaussian_filter(output_path, gaussian_filtered_path)

    # Convert to grayscale and save
    gray_scaled_path = os.path.join(output_gray_dir, f"{date_str}_{time_str}_PS_{processing_type}_{band_count}_gray.tif")
    convert_to_grayscale(gaussian_filtered_path, gray_scaled_path)

    print(f"Image {file} processed. Gaussian filtered and grayscale images saved.")

print("Processing complete.")