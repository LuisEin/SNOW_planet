# -*- coding: utf-8 -*-
'''
Created on Thursday Aug 01 10:23:12 2024

This file takes shaded CBSI data that has been processed in run_07_shadow_split.
Uses these files to create a binary mask of the shaded areas and 
Clips the original 8_Band PS scenes with that mask.
Saves these clipped files to a new directory.


@author: luis
'''

import os
from osgeo import gdal, ogr
import numpy as np

def load_tif_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]

def create_binary_mask_tif(shady_file, mask_dir):
    # Load the TIFF file
    ds = gdal.Open(shady_file)
    if ds is None:
        print(f"Failed to open {shady_file}")
        return None

    # Read the first band (assuming single-band image)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()

    # Check if data contains valid values
    if data is None:
        print(f"No valid data in {shady_file}")
        return None

    # Create a mask: 1 for valid (non-nan) areas, 0 for nan areas
    mask = np.where(np.isnan(data), np.nan, 1)

    # Create the output mask file path with the new naming convention
    base_name = os.path.basename(shady_file).split('_PS_TOAR_8b')[0]
    output_mask_path = os.path.join(mask_dir, f"{base_name}_PS_TOAR_8b_shadow_masked{addon_name}.tif")

    # Save the mask to a new TIFF file with the same resolution and CRS
    driver = gdal.GetDriverByName("GTiff")
    mask_ds = driver.Create(
        output_mask_path,
        ds.RasterXSize,
        ds.RasterYSize,
        1,
        gdal.GDT_Float32,
    )

    # Set the geotransform and projection from the original dataset
    mask_ds.SetGeoTransform(ds.GetGeoTransform())
    mask_ds.SetProjection(ds.GetProjection())

    # Write the mask to the output file
    mask_ds.GetRasterBand(1).WriteArray(mask)

    # Clean up
    mask_ds = None
    ds = None

    print(f"Processed and saved mask file for {shady_file}")
    return output_mask_path

def clip_with_mask(orig_tif, mask_tif, output_filename):
    # Load original tif
    orig_ds = gdal.Open(orig_tif)
    
    # Load mask tif
    mask_ds = gdal.Open(mask_tif)
    mask_band = mask_ds.GetRasterBand(1)
    mask_array = mask_band.ReadAsArray()
    
    # Create a memory file to store the masked image
    mem_driver = gdal.GetDriverByName('MEM')
    mem_ds = mem_driver.Create('', orig_ds.RasterXSize, orig_ds.RasterYSize, orig_ds.RasterCount, gdal.GDT_Float32)
    mem_ds.SetGeoTransform(orig_ds.GetGeoTransform())
    mem_ds.SetProjection(orig_ds.GetProjection())
    
    for i in range(1, orig_ds.RasterCount + 1):
        orig_band = orig_ds.GetRasterBand(i)
        orig_array = orig_band.ReadAsArray()
        
        # Apply the mask
        clipped_array = np.where(mask_array == 1, orig_array, np.nan)
        
        mem_ds.GetRasterBand(i).WriteArray(clipped_array)
    
    # Save the clipped image to the output directory
    output_driver = gdal.GetDriverByName('GTiff')
    output_ds = output_driver.CreateCopy(output_filename, mem_ds, 0)
    
    # Clean up
    orig_ds = None
    mem_ds = None
    mask_ds = None
    output_ds = None

def process_clipping(shady_dir, orig_dir, output_dir, mask_dir):
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    
    shady_files = load_tif_files(shady_dir)
    orig_files = load_tif_files(orig_dir)
    
    # Create a dictionary to map original files based on the date-time prefix
    orig_dict = {}
    for orig_file in orig_files:
        base_name = os.path.basename(orig_file)
        date_time_prefix = "_".join(base_name.split('_')[:2])
        orig_dict[date_time_prefix] = orig_file
    
    for shady_file in shady_files:
        base_name = os.path.basename(shady_file)
        date_time_prefix = "_".join(base_name.split('_')[:2])
        if date_time_prefix in orig_dict:
            orig_file = orig_dict[date_time_prefix]
            mask_tif_path = create_binary_mask_tif(shady_file, mask_dir)
            if mask_tif_path:
                output_filename = os.path.join(output_dir, f"{date_time_prefix}_PS_TOAR_8b{addon_name}.tif")
                clip_with_mask(orig_file, mask_tif_path, output_filename)
        else:
            print(f"Original file for {shady_file} not found.")

if __name__ == "__main__":
    # Define the input directory containing the shady parts TIFF files
    shady_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shade_no_shade_8b_TOAR_gaussian_filtered/shaded_offset_0.02"

    # Define the input directory containing the original images
    orig_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band_gaussian_filtered"

    # Define the output directory where clipped TIFF files will be saved
    output_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shadow_masked_original_ready_files_gaussian_filtered_offset_0.02"

    # Define the directory where mask TIFF files will be saved
    mask_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/mask_files_gaussian_filtered_offset_0.02"

    # gaussian filtered data? -> set True if not -> False
    Gauss = True
    
    if Gauss:
        addon_name = "_gaussian_filtered"
    else: 
        None

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Process the clipping
    process_clipping(shady_dir, orig_dir, output_dir, mask_dir)
