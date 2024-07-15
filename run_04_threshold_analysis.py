# -*- coding: utf-8 -*-
'''
Created on Tuesday Jul 09 09:13:12 2024

This file takes the BlueSnowThreshold calculated PlanetScope data and 
smooths it using a 1-d gaussian filter.
Then does Histograms and threshold calculations.

@author: luis
'''


from functions_planet import *

if __name__ == "__main__":
    # Define the input directory containing the cropped TIFF files
    input_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/indices_8Band_2024_04_16__psscene_analytic_8b_sr_udm2/BST"

    # Define the output directory where histograms will be saved
    output_histogram_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/Indices/BST_Histograms/8_Band"

    # Process the histograms
    process_histograms(input_dir, output_histogram_dir)