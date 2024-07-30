# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 24 09:13:12 2024

This file uses Shadow Index calculated images from the scripts run_02 and run_03
And calculates potential shadow mask

@author: luis
'''


from functions_planet import process_SIindex_directory


    
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
The Result is a file that presents a potential shadow mask
'''
if __name__ == "__main__":
    # Define the input directory containing the SI index files
    input_directory = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/indices_8Band_matching_LIDAR_or_SWS_20240229_20230807_20230302_TOAR_psscene_analytic_8b_udm2/SI_Index"
    
    # Define the output directory where shadow masks will be saved
    output_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/code/temp"

    # Process the SI index files to extract cloud shadows
    process_SIindex_directory(input_directory, output_dir, threshold=0.1)
