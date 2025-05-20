from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import tensorflow as tf
import sys

class DataTransformation:
    def __init__(self):
        self.noise_level=0.1

    def apply_gaussian_noise(self,images):
        """Add Gaussian noise to images and clip to valid range."""
        logging.info("trying to apply gaussian noise")
        try:
            noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=self.noise_level, dtype=tf.float32)
            noisy_images = images + noise
            return tf.clip_by_value(noisy_images, 0.0, 1.0)
        except Exception as e:
            logging.info("failed to apply gaussian noise")
            raise CustomException(e,sys)

    def transform_data_for_fit(self,cover_images, hide_images):
        """transform data into model format"""
        logging.info("converting data into model format")
        try:
            return {"hide_input": hide_images, "cover_input": cover_images}, \
                {"Encoder": cover_images, "Decoder": hide_images}
        except Exception as e:
            logging.info("failed to transform data")
            raise CustomException(e,sys)

