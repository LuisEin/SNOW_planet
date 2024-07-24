# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 09 09:13:12 2024

This file takes the BlueSnowThreshold calculated PlanetScope data and 
smooths it using a 1-d gaussian filter.
Then does Histograms and threshold calculations.

@author: luis
'''


from functions_planet import process_histograms

if __name__ == "__main__":
    # Define the input directory containing the cropped TIFF files
    input_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/indices_8Band_2024_04_26__psscene_analytic_8b_sr_udm2/CBST"

    # Define the output directory where histograms will be saved
    output_histogram_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/CBST_Histograms/8_Band"

    
    # Don't forget to change the index type in the Plot heading between BST and CBST
    index_type = "CBST"
    
    # Process the histograms
    process_histograms(input_dir, index_type, output_histogram_dir)
    

