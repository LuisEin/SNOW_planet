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
from skimage.morphology import reconstruction
import numpy as np
import matplotlib.pyplot as plt
import glob, os, shutil, re, cv2

#### for run_01_merge_tiles ###############################################


# Helper function to mask overlapping pixels
def apply_water_mask(image_path, water_mask_array, water_mask_transform):
    #%%
    ds = gdal.Open(image_path, gdal.GA_Update)
    transform = ds.GetGeoTransform()

    # Compute offsets and sizes
    x_offset = int((water_mask_transform[0] - transform[0]) / transform[1])
    y_offset = int((water_mask_transform[3] - transform[3]) / transform[5])
    
    x_end = x_offset + water_mask_array.shape[1]
    y_end = y_offset + water_mask_array.shape[0]
    
    x_start = max(0, x_offset)
    y_start = max(0, y_offset)
    
    x_end = min(ds.RasterXSize, x_end)
    y_end = min(ds.RasterYSize, y_end)
    
    mask_x_start = max(0, -x_offset)
    mask_y_start = max(0, -y_offset)
    
    mask_x_end = mask_x_start + (x_end - x_start)
    mask_y_end = mask_y_start + (y_end - y_start)

    # Apply the mask to each band
    if x_start < x_end and y_start < y_end:
        for band_idx in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(band_idx)
            array = band.ReadAsArray()
            mask = water_mask_array[mask_y_start:mask_y_end, mask_x_start:mask_x_end] > 0
            nodata_value = band.GetNoDataValue() if band.GetNoDataValue() is not None else 0
            array[y_start:y_end, x_start:x_end][mask] = nodata_value
            band.WriteArray(array)
            band.FlushCache()
    
    ds = None
#%%


# Read AOI from shapefile
def read_shapefile(shapefile_path):
    #%%
    shapefile = ogr.Open(shapefile_path)
    if shapefile is None:
        raise ValueError(f"Could not open shapefile: {shapefile_path}")
    
    layer = shapefile.GetLayer()
    if layer is None:
        raise ValueError(f"Could not get layer from shapefile: {shapefile_path}")
    
    feature = layer.GetFeature(0)
    if feature is None:
        raise ValueError(f"Could not get feature from layer: {shapefile_path}")
    
    geom = feature.GetGeometryRef()
    if geom is None:
        raise ValueError(f"Could not get geometry from feature: {shapefile_path}")
    
    return geom.Clone()
#%%

# Extract time from filename
def extract_time_from_filename(filename):
    #%%
    match = re.search(r'_(\d{6})_', filename)
    if match:
        return match.group(1)
    return ''
#%%

# Check if image covers the AOI
def check_coverage(image_path, aoi_geom):
    ds = gdal.Open(image_path)
    gt = ds.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * ds.RasterXSize
    miny = maxy + gt[5] * ds.RasterYSize
    ds_geom = ogr.CreateGeometryFromWkt(f"POLYGON (({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))")
    print(f"Image Geometry: {ds_geom.ExportToWkt()}")
    return aoi_geom.Within(ds_geom)
#%% end of function

## This is for the OpenCV Version run_01.1 ######
# Apply Gaussian filter
def apply_gaussian_filter(input_path, output_path):
    #%%
    ds = gdal.Open(input_path)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.CreateCopy(output_path, ds, 0)
    
    for band_idx in range(1, out_ds.RasterCount + 1):
        band = out_ds.GetRasterBand(band_idx)
        array = band.ReadAsArray()
        blurred_array = cv2.GaussianBlur(array, (5, 5), 0)
        band.WriteArray(blurred_array)
        band.FlushCache()
    
    ds = None
    out_ds = None
    #%%

# Convert to grayscale - maybe use this function right before the actual analysis
def convert_to_grayscale(input_path, output_path):
    #%%
    ds = gdal.Open(input_path)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    
    array = np.zeros((ds.RasterYSize, ds.RasterXSize), dtype=np.float32)
    for band_idx in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(band_idx)
        band_array = band.ReadAsArray().astype(np.float32)
        array += band_array / ds.RasterCount
    
    gray_array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out_ds.GetRasterBand(1).WriteArray(gray_array)
    out_ds.FlushCache()
    
    ds = None
    out_ds = None
    #%%


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
    
    # Calculate indices based on output_name
    if output_name == "NDVI":
        index = (nir_band - red_band) / (nir_band + red_band)
    elif output_name == "BST":
        index = (nir_band - blue_band) / (nir_band + blue_band)
    elif output_name == "GST":
        index = (nir_band - green_band) / (nir_band + green_band)
    elif output_name == "SI_Index":
        mean_red = np.mean(red_band)
        mean_nir = np.mean(nir_band)
        index = np.sqrt((red_band / mean_red) * (nir_band / mean_nir))
    else:
        raise ValueError("Unknown index: {}".format(output_name))

    # Avoid division by zero for standard indices
    if output_name in ["NDVI", "BST", "GST"]:
        index = np.where((nir_band + red_band) == 0, np.nan, index)
    
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

def do_index_calculation_8band(file, width, output_name, output_dir, product_type):
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

    # Initialize the denominator to avoid unbound local error
    denominator = None
    
    # Calculate indices based on output_name
    if output_name == "NDVI":
        denominator = (nir_band + red_band)
        index = (nir_band - red_band) / denominator
    elif output_name == "BSI":
        denominator = (nir_band + blue_band)
        index = (nir_band - blue_band) / denominator
    elif output_name == "CBSI":
        denominator = (nir_band + coastal_blue_band)
        index = (nir_band - coastal_blue_band) / denominator
    elif output_name == "GSI":
        denominator = (nir_band + green_band)
        index = (nir_band - green_band) / denominator
    elif output_name == "SI_Index":
        mean_red = np.mean(red_band)
        mean_nir = np.mean(nir_band)
        index = np.sqrt((red_band / mean_red) * (nir_band / mean_nir))
    else:
        raise ValueError("Unknown index: {}".format(output_name))

    # Avoid division by zero
    if denominator is not None:
        index = np.where(denominator == 0, np.nan, index)

    # Get georeference info
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # Create the output file for each tile
    driver = gdal.GetDriverByName('GTiff')
    out_file = os.path.join(output_dir, f'{base_name}_{product_type}_{output_name}_width_{width}px.tif')
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



#### from run_04_treshold_analysis_temp #################################################

def apply_gaussian_filter_and_generate_histogram(tiff_file,index_type, output_histogram_dir):
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
    plt.title(f'Histogram of {index_type} from {date}')
    
    # Construct the output file path
    output_file = os.path.join(output_histogram_dir, f'{date}_{index_type}_hist.png')
    
    # Save the histogram
    plt.savefig(output_file)
    plt.close()
    print(f'Histogram saved to {output_file}')

    # Close the dataset
    dataset = None
#%% end of function

def process_histograms(input_dir, index_type, output_histogram_dir):
    #%%
    # This is the wrapper function from apply_gaussian_filter_and_generate_histogram
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.tif'):
                tiff_file = os.path.join(root, file)
                apply_gaussian_filter_and_generate_histogram(tiff_file, index_type, output_histogram_dir)

    
#%% end of function

def fillhole(image):
    #%%
    """
    Apply the morphological fill-hole (flood-fill) transformation.
    With the scikit function from here: 
    https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html
    """
    # Create the marker image
    marker = np.copy(image)
    # Select all rows and columns except the first and the last ones
    marker[1:-1, 1:-1] = image.max()
    
    # Perform morphological reconstruction using skimage
    filled = reconstruction(marker, image, method='erosion')
    
    return filled
#%% end of function

def process_si_index(file_path, output_dir, threshold=0.1):
    #%%
    """
    Process a single SI index file to extract cloud shadows and save the result.
    Reads an SI index file using GDAL.
    Applies the fill-hole transformation to the SI index.
    Computes the difference image and creates a binary shadow mask based on the threshold.
    Extracts the date part from the file name to use in the output file name.
    Saves the resulting shadow mask as a new .tif file using GDAL.
    """
    # Read the SI index file using GDAL
    src_ds = gdal.Open(file_path)
    si_index = src_ds.GetRasterBand(1).ReadAsArray()
    
    # Apply the fillhole transformation to SI
    filled_SI = fillhole(si_index)
    
    # Calculate the difference image
    difference_image = filled_SI - si_index
    
    # Create the potential shadow mask (pSM) using the threshold
    pSM = (difference_image >= threshold).astype(np.uint8)  # Convert to binary mask (0, 1)
    
    # Extract date from the file name (assuming it is the first part of the name)
    base_name = os.path.basename(file_path)
    date_part = base_name.split('_')[0]
    
    # Define the output file path
    output_file = os.path.join(output_dir, f"{date_part}_potential_shadow_mask.tif")
    
    # Save the resulting shadow mask using GDAL
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(output_file, src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())
    dst_ds.GetRasterBand(1).WriteArray(pSM)
    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None
#%% end of function

def process_SIindex_directory(input_dir, output_dir, threshold=0.1):
    #%%
    """
    Process all SI index files in the directory. and calculate a potential
    shadow mask based on the Shadow Index and the morphological fill-hole 
    (flood-fill) transformation.
    From Wang et al., 2021
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".tif"):
            file_path = os.path.join(input_dir, file_name)
            process_si_index(file_path, output_dir, threshold)
#%% end of function
