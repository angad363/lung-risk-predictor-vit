import os
import gdown
import zipfile
from src.lung_cancer_mortality import logger
from src.lung_cancer_mortality.utils.common import get_size
from pathlib import Path
from src.lung_cancer_mortality.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            # Use gdown to download the file from Google Drive
            gdown.download(self.config.source_URL, self.config.local_data_file, quiet=False)
            logger.info(f"{self.config.local_data_file} downloaded!")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            logger.info(f"Extracted zip file to {unzip_path}")