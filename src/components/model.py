from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from tensorflow.keras.layers import Conv2D,Input
from tensorflow.keras import models
import tensorflow as tf
import os
import sys

@dataclass
class ModelConfig:
    input_shape=(224,224,3)
    learning_rate=0.001
    encoder_loss_weight=1.0
    decoder_loss_weight=0.75

class Model:
    def __init__(self):
        self.model_config=ModelConfig()

    def make_encoder(self):
        logging.info("creating encoder model")
        try:
            secret = Input(shape=self.model_config.input_shape, name='secret')
            cover = Input(shape=self.model_config.input_shape, name='cover')

            prepare_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='prep_conv3x3_1')(secret)
            prepare_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='prep_conv3x3_2')(
                prepare_conv_3x3)
            prepare_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='prep_conv3x3_3')(
                prepare_conv_3x3)
            prepare_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='prep_conv3x3_4')(
                prepare_conv_3x3)

            prepare_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='prep_conv4x4_1')(secret)
            prepare_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='prep_conv4x4_2')(
                prepare_conv_4x4)
            prepare_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='prep_conv4x4_3')(
                prepare_conv_4x4)
            prepare_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='prep_conv4x4_4')(
                prepare_conv_4x4)

            prepare_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='prep_conv5x5_1')(secret)
            prepare_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='prep_conv5x5_2')(
                prepare_conv_5x5)
            prepare_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='prep_conv5x5_3')(
                prepare_conv_5x5)
            prepare_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='prep_conv5x5_4')(
                prepare_conv_5x5)

            prepare_concat_1 = tf.concatenate([prepare_conv_3x3, prepare_conv_4x4, prepare_conv_5x5], axis=3,
                                           name="prep_concat_1")

            prepare_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='prep_conv5x5_f')(
                prepare_concat_1)
            prepare_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='prep_conv4x4_f')(
                prepare_concat_1)
            prepare_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='prep_conv3x3_f')(
                prepare_concat_1)

            prepare_prepare_concat_f1 = tf.concatenate([prepare_conv_5x5, prepare_conv_4x4, prepare_conv_3x3], axis=3,
                                                    name="prep_concat_2")

            hide_concat_h = tf.concatenate([cover, prepare_prepare_concat_f1], axis=3, name="hide_concat_1")

            hide_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='hide_conv3x3_1')(
                hide_concat_h)
            hide_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='hide_conv3x3_2')(
                hide_conv_3x3)
            hide_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='hide_conv3x3_3')(
                hide_conv_3x3)
            hide_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='hide_conv3x3_4')(
                hide_conv_3x3)

            hide_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='hide_conv4x4_1')(
                hide_concat_h)
            hide_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='hide_conv4x4_2')(
                hide_conv_4x4)
            hide_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='hide_conv4x4_3')(
                hide_conv_4x4)
            hide_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='hide_conv4x4_4')(
                hide_conv_4x4)

            hide_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='hide_conv5x5_1')(
                hide_concat_h)
            hide_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='hide_conv5x5_2')(
                hide_conv_5x5)
            hide_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='hide_conv5x5_3')(
                hide_conv_5x5)
            hide_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='hide_conv5x5_4')(
                hide_conv_5x5)

            hide_concat_1 = tf.concatenate([hide_conv_3x3, hide_conv_4x4, hide_conv_5x5], axis=3, name="hide_concat_2")

            hide_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='hide_conv5x5_f')(
                hide_concat_1)
            hide_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='hide_conv4x4_f')(
                hide_concat_1)
            hide_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='hide_conv3x3_f')(
                hide_concat_1)

            hide_concat_f1 = tf.concatenate([hide_conv_5x5, hide_conv_4x4, hide_conv_3x3], axis=3, name="hide_concat_3")

            cover_predict = Conv2D(3, kernel_size=1, padding="same", name='hide_conv_f')(hide_concat_f1)
            logging.info("successfully created encoder model")
            return models.Model(inputs=[secret, cover],
                                outputs=cover_predict,
                                name='Encoder')
        except Exception as e:
            logging.info("failed to create encoder model")
            raise CustomException(e,sys)
    def make_decoder(self):
        logging.info("creating decoder")
        try:
            noise_ip = Input(shape=self.model_config.input_shape, name='noise')
            reveal_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='revl_conv3x3_1')(noise_ip)
            reveal_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='revl_conv3x3_2')(
                reveal_conv_3x3)
            reveal_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='revl_conv3x3_3')(
                reveal_conv_3x3)
            reveal_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='revl_conv3x3_4')(
                reveal_conv_3x3)

            reveal_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='revl_conv4x4_1')(noise_ip)
            reveal_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='revl_conv4x4_2')(
                reveal_conv_4x4)
            reveal_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='revl_conv4x4_3')(
                reveal_conv_4x4)
            reveal_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='revl_conv4x4_4')(
                reveal_conv_4x4)

            reveal_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='revl_conv5x5_1')(noise_ip)
            reveal_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='revl_conv5x5_2')(
                reveal_conv_5x5)
            reveal_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='revl_conv5x5_3')(
                reveal_conv_5x5)
            reveal_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='revl_conv5x5_4')(
                reveal_conv_5x5)

            reveal_concat_1 = tf.concatenate([reveal_conv_3x3, reveal_conv_4x4, reveal_conv_5x5], axis=3, name="revl_concat_1")

            reveal_conv_5x5 = Conv2D(50, kernel_size=5, padding="same", activation='relu', name='revl_conv5x5_f')(
                reveal_concat_1)
            reveal_conv_4x4 = Conv2D(50, kernel_size=4, padding="same", activation='relu', name='revl_conv4x4_f')(
                reveal_concat_1)
            reveal_conv_3x3 = Conv2D(50, kernel_size=3, padding="same", activation='relu', name='revl_conv3x3_f')(
                reveal_concat_1)

            reveal_concat_f1 = tf.concatenate([reveal_conv_5x5, reveal_conv_4x4, reveal_conv_3x3], axis=3,
                                           name="revl_concat_2")

            secret_predict = Conv2D(3, kernel_size=1, padding="same", name='revl_conv_f')(reveal_concat_f1)
            logging.info("creating decoder is a success")
            return models.Model(inputs=noise_ip,
                                outputs=secret_predict,
                                name='Decoder')
        except Exception as e:
            logging.info("failed to create decoder")
            raise CustomException(e,sys)

    def make_combined_model(self,encoder,decoder):
        logging.info("creating combined model")
        try:
            combined_input1 = Input(shape=self.model_config.input_shape, name="hide_input")
            combined_input2 = Input(shape=self.model_config.input_shape, name="cover_input")
            watermarked = encoder([combined_input1, combined_input2])
            recovered = decoder(watermarked)
            model = models.Model([combined_input1, combined_input2], [watermarked, recovered], name="full_model")
            logging.info("creating combined model is a success")
            return model
        except Exception as e:
            logging.info("failed to create combined model")
            raise CustomException(e,sys)

