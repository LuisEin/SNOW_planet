# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 10 15:32:12 2024

This script takes an example of merged and clipped tiles from the run_merge_tiles.py
And finds a threshold for shady and non-shady areas

@author: luis
'''

from osgeo import gdal, ogr
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import glob, os, shutil



# Define file paths and date pattern
# BST
input_pattern = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/indices_matching_to_SWS_02_24_until_06-26_psscene_analytic_8b_sr_udm2/BST/20240625_merged_BST_width_3334px.tif' 
# RGB
input_pattern2 = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/indices_matching_to_SWS_02_24_until_06-26_psscene_analytic_8b_sr_udm2/RGB/20240625_merged.tif'
# CBST
input_pattern3 = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/indices_matching_to_SWS_02_24_until_06-26_psscene_analytic_8b_sr_udm2/CBST/20240625_merged_CBST_width_3334px.tif'
# /home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/merged_tiles/BST/20240229_merged_masked_BST_width_3332px.tif
# Adjust the pattern to match your tiles
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/code/temp/'

## Load indexed image 
dataset = gdal.Open(input_pattern3, gdal.GA_ReadOnly)
for x in range(1, dataset.RasterCount + 1):
    band = dataset.GetRasterBand(x)
    array = band.ReadAsArray()

## Filter 
gau_array = gaussian_filter1d(array, 3)

# Get unique values and their counts
unique_values, counts = np.unique(array, return_counts=True)

# Display the unique values and their counts
print("Unique values:", unique_values)
print("Counts:", counts)

# Plot the array using Matplotlib
plt.imshow(gau_array, cmap='gray')
plt.colorbar()
plt.title('CBST gaussian filtered array 2024_06_25')
plt.show()

# Display Histogram
# Plot Histogram
plt.hist(gau_array.flatten(), bins=1000)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of CBST gaussian filtered array 2024_06_25')
plt.ylim(0, 400000)
plt.xlim(-0.2, 1)


# check for no data values
nodata_value = band.GetNoDataValue()
print(f"Nodata-Wert: {nodata_value}")


######## Load RGB 8 Band image ####################################################### 
dataset = gdal.Open(input_pattern2, gdal.GA_ReadOnly)

# Read the bands
bands = []
for i in range(1, 9):
    band = dataset.GetRasterBand(i).ReadAsArray()
    bands.append(band)

# Close the dataset
dataset = None



######### Plot Histograms of all four channels ################################
# Plot histograms
band_names = ['Coastal Blue', 'Blue', 'Green1', 'Green', 'Yellow', 'Red', 'Red Edge', 'NIR']
plt.figure(figsize=(16, 12))

# Adjust text size
plt.rcParams.update({'font.size': 10})

# Find the max pixel value for setting limits
max_pixel_value = max(band.max() for band in bands)

for i, band in enumerate(bands):
    plt.subplot(2, 4, i+1)
    plt.hist(band.flatten(), bins=256, color='gray',alpha=0.7)
    plt.suptitle('Multispectral PlanetScope Image 2024-06-25')
    plt.title(f'{band_names[i]} Band')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.axis([0, max_pixel_value, 0, 700000])  # Set the same axis limits for all subplots


plt.tight_layout()
plt.show()


######### Plot a comparison of BSI with Coastal blue vs blue ##################