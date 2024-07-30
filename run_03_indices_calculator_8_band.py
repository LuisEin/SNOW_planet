#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Jun 25 14:25:12 2024

This script loads 8Band PlanetScope Scenes
Extracts its bands 
Does Indices calculations with various bands
Exports the calculated index as .tif files
While also saving a copy of the clipped RGB image.

Run raw files with run_merge_tiles.py before

@author: luis
"""

from functions_planet import *
import glob, os, shutil

# Define file paths and date pattern
input_pattern = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band/TOAR/*.tif'  
# Adjust the pattern to match your tiles
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/TOAR'
rgb_dir = os.path.join(output_dir, 'RGB')
ndvi_dir = os.path.join(output_dir, 'NDVI')
bsi_dir = os.path.join(output_dir, 'BSI')
cbsi_dir = os.path.join(output_dir, 'CBSI')
gsi_dir = os.path.join(output_dir, 'GSI')
SI_Index_dir = os.path.join(output_dir, 'SI_Index')

# For name string define product type either -> TOAR or SR
product_type = "TOAR"

# List for the desired index calculations:
# NDVI
# BST - blue snow index
# CBST - coastal blue snow index
# GST - green snow index
# SI_Index - snow index
outputs = ["NDVI", "BSI", "CBSI", "GSI", "SI_Index"]

# List for the desired output folders
out_dirs = [ndvi_dir, bsi_dir, cbsi_dir, gsi_dir, SI_Index_dir]

# Create the output directories if they don't exist
os.makedirs(rgb_dir, exist_ok=True)
for dir in out_dirs:
    os.makedirs(dir, exist_ok=True)

# Main loop to iterate over the indices and the output directories
# Filter the files, for only files covering the whole area of interest
# Find all tiles for the specified pattern
tile_files = glob.glob(input_pattern)

# Filter files that are wider than 200 pixels
filtered_files = []
for file in tile_files:
    width, height = get_image_dimensions(file)
    if width > 200 and height > 200:
        filtered_files.append((file, width))
        # Copy file to the RGB directory
        shutil.copy(file, rgb_dir)

# Ensure we have at least one file to process
if not filtered_files:
    raise ValueError("No suitable files found for the specified criteria.")

# Loop through each index and its corresponding directory
for output_name, output_dir in zip(outputs, out_dirs):
    for file, width in filtered_files:
        do_index_calculation_8band(file, width, output_name, output_dir, product_type)
