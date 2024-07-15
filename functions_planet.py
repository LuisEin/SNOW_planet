# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 12 16:13:12 2024

This script lists alle the functions that have been written in the course of 
Analyzing PlanetScope data from Mount Zugspitze Germany.
The general aim is to derive snow cover area

@author: luis
'''
from osgeo import gdal, ogr
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import glob, os, shutil

#### from former run_merge_tiles ###############################################


def read_shapefile(shapefile_path):
    #%%
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()
    geom = layer.GetFeature(0).GetGeometryRef()
    return geom
#%% end of function


#### from former run_indices_calculator.py ####################################

def get_image_dimensions(file_path):
    #%%
    dataset = gdal.Open(file_path)
    if not dataset:
        raise FileNotFoundError(f"Unable to open file: {file_path}")
    
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    dataset = None
    return width, height
#%% end of function

def do_index_calculation_4band(file, width, output_name, output_dir):
    #%%
    '''
    This function needs 4Band surface reflectance data as input
    '''
    dataset = gdal.Open(file)
    
    # Extract the date from the file name (assuming the date is part of the file name in the format YYYYMMDD)
    base_name = os.path.basename(file).split('.')[0]
    
    # Read the bands (assuming the bands are in the order: Blue, Green, Red, NIR)
    blue_band = dataset.GetRasterBand(1).ReadAsArray().astype(float)
    green_band = dataset.GetRasterBand(2).ReadAsArray().astype(float)
    red_band = dataset.GetRasterBand(3).ReadAsArray().astype(float)
    nir_band = dataset.GetRasterBand(4).ReadAsArray().astype(float)
    
    # # Apply water masking
    # water_mask = mask_water(nir_band, green_band, water_threshold)

    # Calculate indices based on output_name
    if output_name == "NDVI":
        index = (nir_band - red_band) / (nir_band + red_band)
    elif output_name == "BST":
        index = (nir_band - blue_band) / (nir_band + blue_band)
    elif output_name == "GST":
        index = (nir_band - green_band) / (nir_band + green_band)
    else:
        raise ValueError("Unknown index: {}".format(output_name))

    # Avoid division by zero
    index = np.where((nir_band + red_band) == 0, np.nan, index)
    
    # # Apply water mask
    # index[water_mask] = np.nan

    # Get georeference info
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # Create the output file for each tile
    driver = gdal.GetDriverByName('GTiff')
    out_file = os.path.join(output_dir, f'{base_name}_{output_name}_width_{width}px.tif')
    out_dataset = driver.Create(out_file, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)

    # Set georeference info to the output file
    out_dataset.SetGeoTransform(geo_transform)
    out_dataset.SetProjection(projection)

    # Write the index band to the output file
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(index)

    # Set NoData value
    out_band.SetNoDataValue(np.nan)

    # Flush data to disk
    out_band.FlushCache()
    out_dataset.FlushCache()

    # Clean up
    del dataset
    del out_dataset

    print(f"{output_name} calculation and export completed for {file}. Output saved as {out_file}")
#%% end of function

def do_index_calculation_8band(file, width, output_name, output_dir):
    #%%
    '''
    This function needs 8Band surface reflectance data as input
    '''
    dataset = gdal.Open(file)
    
    # Extract the date from the file name (assuming the date is part of the file name in the format YYYYMMDD)
    base_name = os.path.basename(file).split('.')[0]
    
    # Read the bands (assuming the bands are in the order: Coastal Blue, Blue, Green I, Green, Yellow, Red, Red Edge, NIR)
    coastal_blue_band = dataset.GetRasterBand(1).ReadAsArray().astype(float)
    blue_band = dataset.GetRasterBand(2).ReadAsArray().astype(float)
    green_1_band = dataset.GetRasterBand(3).ReadAsArray().astype(float)
    green_band = dataset.GetRasterBand(4).ReadAsArray().astype(float)
    yellow_band = dataset.GetRasterBand(5).ReadAsArray().astype(float)
    red_band = dataset.GetRasterBand(6).ReadAsArray().astype(float)
    red_edge_band = dataset.GetRasterBand(7).ReadAsArray().astype(float)
    nir_band = dataset.GetRasterBand(8).ReadAsArray().astype(float)
    
    # Calculate indices based on output_name
    if output_name == "NDVI":
        denominator = (nir_band + red_band)
        index = (nir_band - red_band) / denominator
    elif output_name == "BST":
        denominator = (nir_band + blue_band)
        index = (nir_band - blue_band) / denominator
    elif output_name == "CBST":
        denominator = (nir_band + coastal_blue_band)
        index = (nir_band - coastal_blue_band) / denominator
    elif output_name == "GST":
        denominator = (nir_band + green_band)
        index = (nir_band - green_band) / denominator
    else:
        raise ValueError("Unknown index: {}".format(output_name))

    # Avoid division by zero
    index = np.where(denominator == 0, np.nan, index)

    # Get georeference info
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # Create the output file for each tile
    driver = gdal.GetDriverByName('GTiff')
    out_file = os.path.join(output_dir, f'{base_name}_{output_name}_width_{width}px.tif')
    out_dataset = driver.Create(out_file, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)

    # Set georeference info to the output file
    out_dataset.SetGeoTransform(geo_transform)
    out_dataset.SetProjection(projection)

    # Write the index band to the output file
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(index)

    # Set NoData value
    out_band.SetNoDataValue(np.nan)

    # Flush data to disk
    out_band.FlushCache()
    out_dataset.FlushCache()

    # Clean up
    del dataset
    del out_dataset

    print(f"{output_name} calculation and export completed for {file}. Output saved as {out_file}")
#%% endend of function

def mask_water(nir_band, green_band, threshold):
    #%%
    """
    Masks water areas in the image based on a threshold calculated from green
    and nir bands. Based on Ji, L., Zhang, L., & Wylie, B. (2009). 
    Analysis of dynamic thresholds for the normalized difference water index. 
    Photogrammetric engineering & remote sensing, 75(11), 1307-1317.
    and Ji, L., Zhang, L., & Wylie, B. (2009). Analysis of dynamic thresholds for the normalized difference water index. Photogrammetric engineering & remote sensing, 75(11), 1307-1317.
    
    Parameters:
    red_band (numpy array): Array representing the red band of the image.
    threshold (float): Threshold value to identify water areas.
    
    Returns:
    numpy array: Masked array where water areas are set to NaN.
    
    Deactivated until further notice
    
    """
    water_mask = (green_band - nir_band)/(green_band + nir_band) > threshold
    return water_mask
#%% end of function



#### from former run_treshold_analysis_temp #################################################

def apply_gaussian_filter_and_generate_histogram(tiff_file, output_histogram_dir):
    #%%
    '''
    This function opens multiple planetScope Index calculated .tif files,
    From a directory
    Reads its bands, applies gaussian filter, plots a histogram.
    '''

    # Open the TIFF file
    dataset = gdal.Open(tiff_file)
    if dataset is None:
        print(f"Unable to open {tiff_file}")
        return

    # Read the raster data
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()

    # Apply Gaussian filter
    gau_array = gaussian_filter1d(array, 3)

    # Extract the date from the file name
    base_name = os.path.basename(tiff_file)
    date_str = base_name.split("_")[0]
    date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

    # Plot Histogram
    plt.hist(gau_array.flatten(), bins=50)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of BST from {date}')
    
    # Create the output directory if it doesn't exist
    date_dir = os.path.join(output_histogram_dir, date)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    
    # Construct the output file path
    output_file = os.path.join(date_dir, f'{date}_BST_hist.png')
    
    # Save the histogram
    plt.savefig(output_file)
    plt.close()
    print(f'Histogram saved to {output_file}')

    # Close the dataset
    dataset = None
#%% end of fuction

def process_histograms(input_dir, output_histogram_dir):
    #%%
    # This is the wrapper function from apply_gaussian_filter_and_generate_histogram
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.tif'):
                tiff_file = os.path.join(root, file)
                apply_gaussian_filter_and_generate_histogram(tiff_file, output_histogram_dir)
    
#%% end of function