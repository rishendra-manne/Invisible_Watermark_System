from src.exception import CustomException
from src.logger import logging
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import sys
IMAGE_RESULTS_SAVE_PATH = os.path.join("artifacts","visualResults")
RESULTS_PATH = os.path.join("artifacts","results")

def plot_and_save_results(watermarked_image,recovered_image,cover_image,hide_image):
    try:
        logging.info("plotting results......")
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.title("Cover Image ")
        plt.imshow(cover_image[0])

        plt.subplot(1, 4, 2)
        plt.title("Hide Image")
        plt.imshow(hide_image[0])

        plt.subplot(1, 4, 3)
        plt.title("Watermarked Image")
        plt.imshow(watermarked_image[0])

        plt.subplot(1, 4, 4)
        plt.title("Recovered Watermark")
        plt.imshow(recovered_image[0])

        plt.tight_layout()

        plt.savefig(IMAGE_RESULTS_SAVE_PATH)
        logging.info("plotting results saved in the designated location")
    except Exception as e:
        logging.info("plotting results failed")
        raise CustomException(e,sys)


def save_results(average_psnr, average_accuracy):
    try:
        logging.info("Saving evaluation results...")

        os.makedirs(RESULTS_PATH, exist_ok=True)
        result_file_path = os.path.join(RESULTS_PATH, "evaluation_results.txt")

        with open(result_file_path, "w") as f:
            f.write("Model Evaluation Results\n")
            f.write("=========================\n")
            f.write(f"Average PSNR: {average_psnr:.2f} dB\n")
            f.write(f"Watermark Retrieval Accuracy: {average_accuracy:.2f}%\n")

        logging.info(f"Results saved successfully at {result_file_path}")

    except Exception as e:
        logging.info("Saving results failed")
        raise CustomException(e, sys)



def compute_mean_std(dataset_dir, image_extensions={'.jpg', '.jpeg', '.png'}):
    """
    Compute the mean and std across all images in the dataset directory.
    Returns:
        mean: list of 3 floats
        std: list of 3 floats
    """
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        raise ValueError("No valid image files found in the directory.")

    mean = np.zeros(3)
    std = np.zeros(3)
    pixel_count = 0

    print(f"Processing {len(image_paths)} images...")

    for path in tqdm(image_paths):
        img = Image.open(path).convert('RGB')
        img = np.array(img) / 255.0  # normalize to [0,1]
        if img.ndim != 3 or img.shape[2] != 3:
            continue

        pixel_count += img.shape[0] * img.shape[1]
        mean += img.sum(axis=(0, 1))
        std += (img ** 2).sum(axis=(0, 1))

    mean /= pixel_count
    std = np.sqrt(std / pixel_count - mean ** 2)

    return mean.tolist(), std.tolist()

