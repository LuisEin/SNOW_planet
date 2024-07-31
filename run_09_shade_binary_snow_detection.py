import os
from osgeo import gdal, ogr
import numpy as np

def load_tif_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]

def create_binary_mask_tif(shady_tif, mask_tif_path):
    # Load shady parts tif
    shady_ds = gdal.Open(shady_tif)
    shady_band = shady_ds.GetRasterBand(1)
    shady_array = shady_band.ReadAsArray()
    
    # Create a binary mask of the shady parts: 1 where there's a value, 0 elsewhere
    mask = np.where(shady_array > 0, 1, 0).astype(np.uint8)
    
    # Save the mask to a new TIF file
    driver = gdal.GetDriverByName('GTiff')
    mask_ds = driver.Create(mask_tif_path, shady_ds.RasterXSize, shady_ds.RasterYSize, 1, gdal.GDT_Byte)
    
    if mask_ds is None:
        raise RuntimeError(f"Failed to create mask file: {mask_tif_path}")

    mask_ds.GetRasterBand(1).WriteArray(mask)
    mask_ds.SetGeoTransform(shady_ds.GetGeoTransform())
    mask_ds.SetProjection(shady_ds.GetProjection())
    
    shady_ds = None
    mask_ds = None

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
            mask_tif_path = os.path.join(mask_dir, f"{date_time_prefix}_mask.tif")
            output_filename = os.path.join(output_dir, f"{date_time_prefix}_PS_TOAR_8b.tif")
            
            # Create the binary mask TIF file
            create_binary_mask_tif(shady_file, mask_tif_path)
            
            # Clip the original image using the mask TIF file
            clip_with_mask(orig_file, mask_tif_path, output_filename)
        else:
            print(f"Original file for {shady_file} not found.")

if __name__ == "__main__":
    # Define the input directory containing the shady parts TIFF files
    shady_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shade_no_shade_8b_TOAR/shaded"

    # Define the input directory containing the original images
    orig_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/files_ready/8_band/TOAR"

    # Define the output directory where clipped TIFF files will be saved
    output_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/shadow_masked_original_ready_files"

    # Define the directory where mask TIFF files will be saved
    mask_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Shadow_mask/mask_files"

    # Process the clipping
    process_clipping(shady_dir, orig_dir, output_dir, mask_dir)
