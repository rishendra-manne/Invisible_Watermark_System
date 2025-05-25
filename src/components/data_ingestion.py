from src.exception import CustomException
from src.logger import logging
from huggingface_hub import snapshot_download
from dataclasses import dataclass
import sys
import os

@dataclass
class IngestionConfig:
    cover_data = os.path.join("artifacts","data","cover_data")
    hide_data = os.path.join("artifacts","data","hide_data")
    output_dir = os.path.join('artifacts', 'data')

class DataIngestion:

    def __init__(self,repo_id:str,token:str):
        self.ingestion_config=IngestionConfig()
        self.repo_id=repo_id
        self.access_token=token

    def get_data(self):
        """

        :return: returns the tuple containing path of cover data and hide data
        """
        logging.info("starting to download the data from hugging face datasets")
        try:
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.access_token,  # required for private datasets
                local_dir=self.config.output_dir
            )
            logging.info("dataset download successful")
            return (self.ingestion_config.cover_data,self.ingestion_config.hide_data)

        except Exception as e:
            logging.info("failed to download the data")
            raise CustomException(e,sys)



