import cv2
import numpy as np
from evaluation import compare_images
import os

# Directories containing the original and reconstructed images
ORIGINAL_DIR = r"C:\Users\91636\Documents\Sem5\CV\Project\data"  # Update this path based on your setup
RECONSTRUCTED_DIR = r"C:\Users\91636\Documents\Sem5\CV\Project\reconstructed_molecules"

# Lists to store SSIM and MSE values for all images
ssim_values = []
mse_values = []

# Loop through all the original images
for image_name in os.listdir(ORIGINAL_DIR):
    original_path = os.path.join(ORIGINAL_DIR, image_name)
    reconstructed_path = os.path.join(RECONSTRUCTED_DIR, "reconstructed_" + image_name)

    # Read the images
    original_image = cv2.imread(original_path)
    reconstructed_image = cv2.imread(reconstructed_path)

    # Compare the images
    ssim_value, mse_value = compare_images(original_image, reconstructed_image)

    ssim_values.append(ssim_value)
    mse_values.append(mse_value)

# Compute average SSIM and MSE
avg_ssim = np.mean(ssim_values)
avg_mse = np.mean(mse_values)

print(f"Average SSIM: {avg_ssim}")
print(f"Average MSE: {avg_mse}")
