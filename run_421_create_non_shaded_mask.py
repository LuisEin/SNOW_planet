#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:50:32 2024

@author: luis
"""

import os
import numpy as np
from osgeo import gdal

# Directories
input_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/non-shaded_mask/mask_files_gaussian_filtered_offset_0.02'
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/non-shaded_mask/non_shaded_mask_gaussian_filtered_offset_0.02'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through each file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.tif'):
        # Extract the date part from the filename
        date_part = filename.split('_')[0]
        
        # Construct the new filename
        new_filename = f"{date_part}_PS_non_shaded_mask.tif"
        new_filepath = os.path.join(output_dir, new_filename)
        
        # Open the dataset
        filepath = os.path.join(input_dir, filename)
        dataset = gdal.Open(filepath, gdal.GA_Update)
        
        if dataset is None:
            print(f"Failed to open {filepath}")
            continue
        
        # Read the raster band (assuming it's a single-band raster)
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray()
        
        # Set the values of 0 to NaN
        array = np.where(array == 0, np.nan, array)
        
        # Create a new output file with the desired filename
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(new_filepath, dataset.RasterXSize, dataset.RasterYSize, 1, band.DataType)
        
        # Set the geotransform and projection from the input dataset
        out_dataset.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset.SetProjection(dataset.GetProjection())
        
        # Write the modified array to the new file
        out_band = out_dataset.GetRasterBand(1)
        out_band.WriteArray(array)
        
        # Set the NoData value to NaN (this is a floating-point NaN)
        out_band.SetNoDataValue(np.nan)
        
        # Flush the cache to write to the file
        out_band.FlushCache()
        
        # Close the datasets
        dataset = None
        out_dataset = None

        print(f"Processed {filename} -> {new_filename}")
