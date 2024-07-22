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
input_pattern = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_merged/8Band_2024_04_16__psscene_analytic_8b_sr_udm2/*merged.tif'  
# Adjust the pattern to match your tiles
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/code/temp/'
rgb_dir = os.path.join(output_dir, 'RGB')
ndvi_dir = os.path.join(output_dir, 'NDVI')
bst_dir = os.path.join(output_dir, 'BST')
cbst_dir = os.path.join(output_dir, 'CBST')
gst_dir = os.path.join(output_dir, 'GST')

# List for the desired index calculations:
# NDVI
# BST - blue snow threshold
# CBST - coastal blue snow threshold
# GST - green snow threshold
outputs = ["NDVI", "BST", "CBST", "GST"]

# List for the desired output folders
out_dirs = [ndvi_dir, bst_dir, cbst_dir, gst_dir]

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
        do_index_calculation_8band(file, width, output_name, output_dir)