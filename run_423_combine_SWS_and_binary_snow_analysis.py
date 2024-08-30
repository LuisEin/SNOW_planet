import numpy as np
from osgeo import gdal
from skimage.morphology import dilation, square
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import jaccard
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

# Paths to the input files
ps_file = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/PlanetScope_Data/RF/binary_classification/combined/20240229_combined_RF_predicted_binary_snow.tif"
sws_file = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Sentinel_Data/SWS/SWS_all_data_processed/binary_1wet_0dry/SWS_2024_02_28_17_15.tif"

# Output directory
output_dir = "/home/luis/Data/04_Uni/03_Master_Thesis/SNOW/02_data/Planet_Sentinel_combined/analysis_metrics"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to read GeoTIFF files using GDAL and handle NoData values
def read_tif(file_path):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    
    # Convert the array to float to handle NaN values
    array = array.astype(float)
    
    # Replace NoData values (255) with NaN
    array[array == 255] = np.nan
    
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    return array, geotransform, projection

# Read PS and SWS data
ps_arr, ps_geo, ps_proj = read_tif(ps_file)
sws_arr, sws_geo, sws_proj = read_tif(sws_file)

# Resample SWS to match PS resolution using np.kron
sws_resampled_arr = np.kron(sws_arr, np.ones((20, 20)))

# Ensure the resampled SWS array has the same shape as the PS array
min_rows = min(ps_arr.shape[0], sws_resampled_arr.shape[0])
min_cols = min(ps_arr.shape[1], sws_resampled_arr.shape[1])
ps_arr = ps_arr[:min_rows, :min_cols]
sws_resampled_arr = sws_resampled_arr[:min_rows, :min_cols]

# Mask out NoData values
nodata_mask = np.isnan(ps_arr) | np.isnan(sws_resampled_arr)
ps_arr[nodata_mask] = np.nan
sws_resampled_arr[nodata_mask] = np.nan

# Buffering Method: Dilate PS snow pixels to account for resolution differences
ps_dilated_arr = dilation(ps_arr == 1, square(20))  # Dilate by 20 pixels (~60m)

# Majority Voting Method: Apply majority voting within each SWS pixel (60% threshold)
majority_vote_arr = np.zeros_like(ps_arr)
for i in range(0, sws_resampled_arr.shape[0], 20):
    for j in range(0, sws_resampled_arr.shape[1], 20):
        block = ps_arr[i:i+20, j:j+20]
        if block.size > 0 and np.nanmean(block) >= 0.6:
            majority_vote_arr[i:i+20, j:j+20] = 1

# Focus only on the pixels where SWS indicates wet snow (1)
wet_snow_mask = sws_resampled_arr == 1

# Extract pixels for the three methods
ps_wet_snow = ps_arr[wet_snow_mask]
ps_wet_snow_dilated = ps_dilated_arr[wet_snow_mask]
ps_wet_snow_majority = majority_vote_arr[wet_snow_mask]
sws_wet_snow = sws_resampled_arr[wet_snow_mask]

# Calculate metrics for the three methods
def calculate_metrics(ps_wet_snow, sws_wet_snow):
    cm = confusion_matrix(ps_wet_snow, sws_wet_snow, labels=[0, 1])
    accuracy = accuracy_score(ps_wet_snow, sws_wet_snow)
    precision = precision_score(ps_wet_snow, sws_wet_snow, zero_division=0)
    recall = recall_score(ps_wet_snow, sws_wet_snow, zero_division=0)
    f1 = f1_score(ps_wet_snow, sws_wet_snow, zero_division=0)
    iou = 1 - jaccard(ps_wet_snow, sws_wet_snow)
    correlation = pearsonr(ps_wet_snow, sws_wet_snow)[0]
    return cm, accuracy, precision, recall, f1, iou, correlation

# Metrics for the original PS data
metrics_original = calculate_metrics(ps_wet_snow, sws_wet_snow)

# Metrics for the buffering method
metrics_dilated = calculate_metrics(ps_wet_snow_dilated, sws_wet_snow)

# Metrics for the majority voting method
metrics_majority = calculate_metrics(ps_wet_snow_majority, sws_wet_snow)

# Save metrics to a text file
metrics_output_path = os.path.join(output_dir, "metrics_all_methods.txt")
with open(metrics_output_path, "w") as f:
    f.write("Original PS Snow Classification Metrics:\n")
    f.write(f"Confusion Matrix:\n{metrics_original[0]}\n")
    f.write(f"Accuracy: {metrics_original[1]:.4f}\n")
    f.write(f"Precision: {metrics_original[2]:.4f}\n")
    f.write(f"Recall: {metrics_original[3]:.4f}\n")
    f.write(f"F1 Score: {metrics_original[4]:.4f}\n")
    f.write(f"Intersection over Union (IoU): {metrics_original[5]:.4f}\n")
    f.write(f"Correlation: {metrics_original[6]:.4f}\n\n")

    f.write("Buffering Method Metrics:\n")
    f.write(f"Confusion Matrix:\n{metrics_dilated[0]}\n")
    f.write(f"Accuracy: {metrics_dilated[1]:.4f}\n")
    f.write(f"Precision: {metrics_dilated[2]:.4f}\n")
    f.write(f"Recall: {metrics_dilated[3]:.4f}\n")
    f.write(f"F1 Score: {metrics_dilated[4]:.4f}\n")
    f.write(f"Intersection over Union (IoU): {metrics_dilated[5]:.4f}\n")
    f.write(f"Correlation: {metrics_dilated[6]:.4f}\n\n")

    f.write("Majority Voting Method Metrics:\n")
    f.write(f"Confusion Matrix:\n{metrics_majority[0]}\n")
    f.write(f"Accuracy: {metrics_majority[1]:.4f}\n")
    f.write(f"Precision: {metrics_majority[2]:.4f}\n")
    f.write(f"Recall: {metrics_majority[3]:.4f}\n")
    f.write(f"F1 Score: {metrics_majority[4]:.4f}\n")
    f.write(f"Intersection over Union (IoU): {metrics_majority[5]:.4f}\n")
    f.write(f"Correlation: {metrics_majority[6]:.4f}\n")

print(f"Metrics saved to {metrics_output_path}")

# Plotting the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title('PS Snow Classification (Original)')
plt.imshow(ps_arr, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(label='Snow (1) / No Snow (0)')

plt.subplot(2, 2, 2)
plt.title('SWS Wet Snow Resampled')
plt.imshow(sws_resampled_arr, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Wet Snow (1) / Dry Snow or No Snow (0)')

plt.subplot(2, 2, 3)
plt.title('PS Snow (Buffered/Dilated)')
plt.imshow(ps_dilated_arr, cmap='Purples', vmin=0, vmax=1)
plt.colorbar(label='Buffered Snow (1) / No Snow (0)')

plt.subplot(2, 2, 4)
plt.title('PS Snow (Majority Voting)')
plt.imshow(majority_vote_arr, cmap='Greens', vmin=0, vmax=1)
plt.colorbar(label='Majority Vote Snow (1) / No Snow (0)')

plot_output_path = os.path.join(output_dir, "comparison_plot.png")
plt.savefig(plot_output_path)
plt.show()

print(f"Plot saved to {plot_output_path}")
