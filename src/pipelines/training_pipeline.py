from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformation
from src.components.model_evalution import ModelEvalution
from src.utils import plot_and_save_results,save_results
from src.components.model import Model
from src.components import model_evalution
import tensorflow as tf
import os
import sys

@dataclass
class TrainingConfig:
    epochs :int = 100
    batch_size :int = 32
    loss=['mse','mse']
    loss_weights=[1.0,0.75]
    learning_rate=0.001
    repo_id="rishi12111/watermarks"
    access_token = os.getenv("HF_ACCESS_TOKEN")
    access_token = "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"

class VisualizeCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset):
        super(VisualizeCallback, self).__init__()
        self.dataset = dataset.take(1).cache()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 1 == 0:
            for cover_images, hide_images in self.dataset:
                watermarked_images, recovered_images = self.model.predict([cover_images, hide_images])
                plot_and_save_results(watermarked_images,recovered_images)


class TrainingPipeline:

    def __init__(self):
        self.training_config=TrainingConfig()
        self.data_ingestion=DataIngestion(self.training_config.repo_id,self.training_config.access_token)
        self.data_transformation=DataTransformation()
        self.data_preprocessing=DataPreprocessing()
        self.model_evalution=ModelEvalution()
        self.model = Model()

    def train_model(self):
        try:
            logging.info("model training initiated")
            cover_data,hide_data= self.data_ingestion.get_data()
            cover_image_paths,hide_image_paths=self.data_preprocessing.get_image_paths((cover_data,hide_data))
            training_set=self.data_preprocessing.load_and_combine_data(cover_image_paths,hide_image_paths)
            fitted_set=training_set.map(self.data_transformation.transform_data_for_fit)
            model=self.model.make_combined_model()
            model.compile(
               optimizer=tf.keras.optimizers.Adam(self.training_config.learning_rate),
               loss=self.training_config.loss,
               loss_weights=self.training_config.loss_weights
              )
            visualize_callback = VisualizeCallback(training_set)

            history = model.fit(
               fitted_set,
               epochs=self.training_config.epochs,
               callbacks=[visualize_callback]
              )
            logging.info("model training finished")
            return model_evalution.evaluate_model(model,training_set)
        except Exception as e:
            logging.info("model training failed")
            raise CustomException(e,sys)


def main():
    pipeline=TrainingPipeline()
    result=pipeline.train_model()
    save_results(result)

if __name__ == "__main__":
    main()








