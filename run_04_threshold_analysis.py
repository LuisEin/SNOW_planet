# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 09 09:13:12 2024

This file takes the BlueSnowThreshold calculated PlanetScope data and 
smooths it using a 1-d gaussian filter.
Then does Histograms and threshold calculations.

@author: luis
'''


from functions_planet import process_histograms, process_SIindex_directory

if __name__ == "__main__":
    # Define the input directory containing the cropped TIFF files
    input_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/indices_8Band_2024_04_16__psscene_analytic_8b_sr_udm2/BST"

    # Define the output directory where histograms will be saved
    output_histogram_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/BST_Histograms/8_Band"

    # Process the histograms
    process_histograms(input_dir, output_histogram_dir)
    
    
### Calculate Shadow-Index (SI) ###############################################
'''
This process has been integrated into the indices calculations (run_02 and run_03)
'''

### Potential Shadow Mask from Wang et al. 2021 ###############################
'''
This function Implements the morphological fill-hole transformation.
Reads an SI index file using GDAL.
Applies the fill-hole transformation to the SI index.
Computes the difference image and creates a binary shadow mask based on the threshold.
Extracts the date part from the file name to use in the output file name.
Saves the resulting shadow mask as a new .tif file using GDAL.
'''
if __name__ == "__main__":
    # Define the input directory containing the SI index files
    input_directory = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/code/temp/SI_Index"
    
    # Define the output directory where shadow masks will be saved
    output_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/code/temp"

    # Process the SI index files to extract cloud shadows
    process_SIindex_directory(input_dir, output_dir, threshold=0.1)
