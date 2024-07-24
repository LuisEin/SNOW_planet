# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 09 17:13:12 2024

This file takes PlanetScope Scenes by the same date and puts the single tiles 
together.
Saves them as one tif file containing the whole AOI.

This is the first script to use when starting an analysis of PlanetScope data

@author: luis
'''

import os
import glob
import numpy as np
from osgeo import gdal, ogr
import shutil
import re

# Define paths
# 8 Band: *AnalyticMS_SR_8b_clip.tif
input_pattern = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_raw/4_band/Data_March-June_23_Feb-March_24_psscene_analytic_sr_udm2/PSScene/*AnalyticMS_SR_clip.tif'
output_dir = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/4_band/Data_March-June_23_Feb-March_24_psscene_analytic_sr_udm2'
shapefile_path = '/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Shapefiles/shapefile_Zugspitze/03_AOI_shp_zugspitze_reproj_for_code/AOI_zugspitze_reproj_32632.shp'  


# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Collect files matching the pattern
files = glob.glob(input_pattern)

# Group files by date and time
files_by_datetime = {}
for file in files:
    basename = os.path.basename(file)
    date_str = basename[:8]
    time_str = basename[9:15]  # Assuming time is in HHMMSS format
    band_info = basename.split('_')[-2]  # Assuming band info is in the last part before 'clip'
    datetime_str = f"{date_str}_{time_str}_{band_info}"
    if datetime_str not in files_by_datetime:
        files_by_datetime[datetime_str] = []
    files_by_datetime[datetime_str].append(file)

# Read AOI from shapefile
def read_shapefile(shapefile_path):
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

try:
    aoi_geom = read_shapefile(shapefile_path)
    print(f"AOI Geometry: {aoi_geom.ExportToWkt()}")
except ValueError as e:
    print(e)
    raise

# Check if image covers the AOI or is larger than the AOI
def check_coverage(image_path, aoi_geom):
    ds = gdal.Open(image_path)
    gt = ds.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * ds.RasterXSize
    miny = maxy + gt[5] * ds.RasterYSize
    ds_geom = ogr.CreateGeometryFromWkt(f"POLYGON (({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))")
    print(f"Image Geometry: {ds_geom.ExportToWkt()}")
    return aoi_geom.Within(ds_geom) or ds_geom.Within(aoi_geom)

# Process each datetime group
for datetime_str, file_list in files_by_datetime.items():
    print(f"Processing datetime: {datetime_str}")

    # Sort files by size (largest first)
    file_list.sort(key=os.path.getsize, reverse=True)

    # Extract date, time, and band information
    date_str, time_str, band_info = datetime_str.split('_')

    # Check if the largest image covers or is larger than the AOI
    if check_coverage(file_list[0], aoi_geom):
        # The largest image covers or is larger than the AOI, copy it to output
        output_path = os.path.join(output_dir, f"{date_str}_{time_str}_PS_ready_{band_info}.tif")
        shutil.copyfile(file_list[0], output_path)
        print(f"Single image {file_list[0]} covers or is larger than the AOI. Copied to {output_path}")
        continue

    # Otherwise, clip the images to fill the AOI
    for i, file in enumerate(file_list):
        output_path = os.path.join(output_dir, f"{date_str}_{time_str}_clip_{i}_PS_ready_{band_info}.tif")
        gdal.Warp(output_path, file, cutlineDSName=shapefile_path, cropToCutline=True, dstNodata=np.nan)
        print(f"Clipped image {file} to AOI and saved to {output_path}")

print("Processing complete.")