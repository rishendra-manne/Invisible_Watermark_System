from src.exception import CustomException
from src.logger import logging
from tensorflow.keras.models import load_model
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
import sys
import os

@dataclass
class PredictionConfig:
    model_path: str = os.path.join("artifacts", "models", "model.h5")
    encoder_path: str = os.path.join("artifacts", "models", "encoder.h5")
    decoder_path: str = os.path.join("artifacts", "models", "decoder.h5")


class PredictionPipeline:

    def __init__(self):
        self.prediction_config = PredictionConfig()
        self.encoder = load_model(self.prediction_config.encoder_path, compile=False)
        self.decoder = load_model(self.prediction_config.decoder_path, compile=False)
        self.data_transform=DataTransformation()

    def hide_watermark(self, cover_image, hide_image):
        try:
            logging.info("Trying to encode the image...")
            cover_image,hide_image=self.data_transform.normalize_batch(cover_image),self.data_transform.normalize_batch(hide_image)
            encoded_image = self.encoder.predict([cover_image, hide_image], verbose=0)
            logging.info("Encoding successful.")
            return self.data_transform.denormalize_batch(encoded_image)
        except Exception as e:
            logging.exception("Failed to encode the image.")
            raise CustomException(e, sys)

    def reveal_watermark(self, watermarked_image):
        try:
            logging.info("Extraction of watermark is in progress...")
            watermarked_image=self.data_transform.normalize_batch(watermarked_image)
            decoded_watermark = self.decoder.predict(watermarked_image, verbose=0)
            logging.info("Extraction of watermark succeeded.")
            return self.data_transform.denormalize_batch(decoded_watermark)
        except Exception as e:
            logging.exception("Extraction of watermark failed.")
            raise CustomException(e, sys)
