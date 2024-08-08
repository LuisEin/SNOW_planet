# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 09 17:13:12 2024

This script processes PlanetScope Scenes by date and applies Gaussian filtering.
The processed images are saved. 

This script is the second, but optional step in analyzing PlanetScope data.

Steps:
1. Load the pre-processed PlanetScope images.
2. Apply Gaussian filtering to smooth the images.
3. Save the processed images in directory.

Note: The input files are assumed to be already clipped and water masked.

@author: Luis
'''

import glob
import os
from functions_planet import extract_time_from_filename, apply_gaussian_filter #, convert_to_grayscale

# Define paths
input_pattern = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band/TOAR/*_ready.tif'
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band_gaussian_filtered'
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
    files_by_date[date_str] = file

# Process each file by date
for date_str, file in files_by_date.items():
    print(f"Processing date: {date_str}")

    # Apply Gaussian filter and save
    time_str = extract_time_from_filename(file)
    gaussian_filtered_path = os.path.join(output_dir, f"{date_str}_{time_str}_PS_{processing_type}_{band_count}_gaussian.tif")
    apply_gaussian_filter(file, gaussian_filtered_path)

    print(f"Image {file} processed. Gaussian filtered image saved as {gaussian_filtered_path}")

print("Processing complete.")