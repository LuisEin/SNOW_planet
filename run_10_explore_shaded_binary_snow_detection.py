# -*- coding: utf-8 -*-

'''
Created on Thursday Aug 01 10:23:12 2024

This file takes the shaded parts of the original ready files and calculates 
Thresholds and saves specific bands.
It was used to find out which approach to use on shadow binary snow detection.
But since the Coastal Blue single Band was chosen, this file is not really 
necessary, or of great use anymore.


@author: luis
'''

import os
import numpy as np
from osgeo import gdal

# Define paths
input_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shadow_masked_original_ready_files'
output_base_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/indices_and_single_bands'

# Create output directories
coastal_blue_dir = os.path.join(output_base_dir, 'coastal_blue')
yellow_dir = os.path.join(output_base_dir, 'yellow')
green_1_band_dir = os.path.join(output_base_dir, 'green_1_band')
cbsi_dir = os.path.join(output_base_dir, 'CBSI')
cbysi_dir = os.path.join(output_base_dir, 'CBYSI')

os.makedirs(coastal_blue_dir, exist_ok=True)
os.makedirs(yellow_dir, exist_ok=True)
os.makedirs(green_1_band_dir, exist_ok=True)
os.makedirs(cbsi_dir, exist_ok=True)
os.makedirs(cbysi_dir, exist_ok=True)

# List of files to process
files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

# Function to save a single band
def save_band(band_array, geo_transform, projection, output_file):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_file, band_array.shape[1], band_array.shape[0], 1, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(geo_transform)
    out_dataset.SetProjection(projection)
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(band_array)
    out_band.SetNoDataValue(np.nan)
    out_band.FlushCache()
    out_dataset.FlushCache()
    del out_dataset

for file in files:
    dataset = gdal.Open(file)
    base_name = os.path.basename(file).split('.')[0]

    # Read the bands
    coastal_blue_band = dataset.GetRasterBand(1).ReadAsArray().astype(float)
    blue_band = dataset.GetRasterBand(2).ReadAsArray().astype(float)
    green_1_band = dataset.GetRasterBand(3).ReadAsArray().astype(float)
    green_band = dataset.GetRasterBand(4).ReadAsArray().astype(float)
    yellow_band = dataset.GetRasterBand(5).ReadAsArray().astype(float)
    red_band = dataset.GetRasterBand(6).ReadAsArray().astype(float)
    red_edge_band = dataset.GetRasterBand(7).ReadAsArray().astype(float)
    nir_band = dataset.GetRasterBand(8).ReadAsArray().astype(float)

    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # Save single bands
    save_band(coastal_blue_band, geo_transform, projection, os.path.join(coastal_blue_dir, f'{base_name}_coastal_blue.tif'))
    save_band(yellow_band, geo_transform, projection, os.path.join(yellow_dir, f'{base_name}_yellow.tif'))
    save_band(green_1_band, geo_transform, projection, os.path.join(green_1_band_dir, f'{base_name}_green_1_band.tif'))

    # Calculate CBSI
    cbsi = (nir_band - coastal_blue_band) / (nir_band + coastal_blue_band)
    cbsi[np.isinf(cbsi)] = np.nan
    save_band(cbsi, geo_transform, projection, os.path.join(cbsi_dir, f'{base_name}_CBSI.tif'))

    # Calculate CBYSI
    cbysi = (yellow_band - coastal_blue_band) / (yellow_band + coastal_blue_band)
    cbysi[np.isinf(cbysi)] = np.nan
    save_band(cbysi, geo_transform, projection, os.path.join(cbysi_dir, f'{base_name}_CBYSI.tif'))

    del dataset

print("Processing complete. All bands and indices have been saved.")
