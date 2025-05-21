from src.exception import CustomException
from src.logger import logging
import matplotlib.pyplot as plt
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




