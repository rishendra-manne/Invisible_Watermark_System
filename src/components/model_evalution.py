from src.exception import CustomException
from src.logger import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.components.data_transformation import DataTransformation
import sys
import os
from skimage.metrics import peak_signal_noise_ratio as psnratio


class ModelEvaluation:

    def __init__(self):
        self.threshold = 0.3

    def evaluate_model(self, model, data):

        try:
            logging.info("Model evaluation starting...")
            total_psnr = 0
            total_acc = 0
            total_samples = 0
            data_transform = DataTransformation()

            for cover_images, hide_images in data:
                watermarked_images, recovered_images = model.predict([cover_images, hide_images])

                # Denormalize all images
                cover_images = data_transform.denormalize_batch(cover_images)
                hide_images = data_transform.denormalize_batch(hide_images)
                watermarked_images = data_transform.denormalize_batch(watermarked_images)
                recovered_images = data_transform.denormalize_batch(recovered_images)

                batch_size = cover_images.shape[0]
                total_samples += batch_size

                # Compute PSNR
                for i in range(batch_size):
                    psnr = psnratio(cover_images[i], watermarked_images[i])
                    total_psnr += psnr

                # Compute pixel-wise accuracy
                diff = tf.abs(recovered_images - hide_images)
                matches = tf.reduce_sum(tf.cast(diff < self.threshold, tf.float32)).numpy()
                total_pixels = np.prod(recovered_images.shape)
                batch_accuracy = (matches / total_pixels) * 100
                total_acc += batch_accuracy

            avg_psnr = total_psnr / total_samples
            retrieval_acc = total_acc / (total_samples)

            logging.info("Model evaluation succeeded")
            return avg_psnr, retrieval_acc

        except Exception as e:
            logging.info("Model evaluation failed")
            raise CustomException(e, sys)


