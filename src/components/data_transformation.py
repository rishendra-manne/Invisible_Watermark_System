from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import compute_mean_std
import tensorflow as tf
import numpy as np
import sys
import os
@dataclass
class DataTransformationConfig:
    mean,std=compute_mean_std(os.path.join("artifacts","data","cover_data"))


class DataTransformation:
    def __init__(self, noise_level=0.1):
        """
        mean, std: lists or arrays of length 3 for RGB channels.
        If None, defaults to ImageNet stats.
        """
        self.config=DataTransformationConfig()
        self.noise_level = noise_level
        self.mean = np.array(self.config.mean )
        self.std = np.array(self.config.std )

    def normalize_batch(self, images):
        """Normalize batch of images with stored mean and std"""
        try:
            logging.info("Normalizing batch of images")
            images = (images - self.mean) / self.std
            return images
        except Exception as e:
            logging.info("Failed to normalize batch")
            raise CustomException(e, sys)

    def denormalize_batch(self, images, should_clip=True):
        """Denormalize batch of images with stored mean and std"""
        try:
            logging.info("Denormalizing batch of images")
            images = (images * self.std) + self.mean
            if should_clip:
                images = np.clip(images, 0, 1)
            return images
        except Exception as e:
            logging.info("Failed to denormalize batch")
            raise CustomException(e, sys)

    def apply_gaussian_noise(self, images):
        """Add Gaussian noise to images and clip to valid range."""
        logging.info("Trying to apply Gaussian noise")
        try:
            noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=self.noise_level, dtype=tf.float32)
            noisy_images = images + noise
            return tf.clip_by_value(noisy_images, 0.0, 1.0)
        except Exception as e:
            logging.info("Failed to apply Gaussian noise")
            raise CustomException(e, sys)

    def transform_data_for_fit(self, cover_images, hide_images):
        """Transform data into model format"""
        logging.info("Converting data into model format")
        try:
            # Normalize input images before feeding into model
            normalized_hide = self.normalize_batch(hide_images)
            normalized_cover = self.normalize_batch(cover_images)
            return {"hide_input": normalized_hide, "cover_input": normalized_cover}, \
                   {"Encoder": cover_images, "Decoder": hide_images}
        except Exception as e:
            logging.info("Failed to transform data")
            raise CustomException(e, sys)
