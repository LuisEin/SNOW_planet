#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:25:12 2024

This script loads 4Band PlanetScope Scenes
Extracts its bands 
Does Indices calculations with various bands
Exports the calculated index as .tif files
While also saving a copy of the clipped RGB image.

Run raw files with run_merge_tiles.py before


@author: luis
"""

import glob, os, shutil
from functions_planet import *


# input_pattern = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/4_band/Data_March-June_23_Feb-March_24_psscene_analytic_sr_udm2/PSScene/*AnalyticMS_SR_clip.tif'  
input_pattern = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_merged/8Band_2024_04_16__psscene_analytic_8b_sr_udm2/*_merged_masked.tif"
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/code/temp/'
rgb_dir = os.path.join(output_dir, 'RGB')
ndvi_dir = os.path.join(output_dir, 'NDVI')
bst_dir = os.path.join(output_dir, 'BST')
gst_dir = os.path.join(output_dir, 'GST')


# List for the desired index calculations:
    # RGB
    # NDVI
    # BST - blue snow threshold
    # GST - green snow threshold
outputs = ["NDVI", "BST", "GST"]

# List for the desired output folders
out_dirs = [ndvi_dir, bst_dir, gst_dir]


# # Water masking threshold
# water_threshold = 0.0  # As 


# Create the output directories if they don't exist
os.makedirs(rgb_dir, exist_ok=True)
for dir in out_dirs:
    os.makedirs(dir, exist_ok=True)

# Main loop to iterate over the indices and the output directories

# filter the files, for only files covering the whole area of interest
    # Find all tiles for the specified pattern
    tile_files = glob.glob(input_pattern)
    
    # Filter files that cover the whole area and are wider than 200 pixels
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
        do_index_calculation_4band(file, width, output_name, output_dir)

