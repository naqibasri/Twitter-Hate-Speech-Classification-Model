import os
from datetime import datetime

# Common constants
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
#BUCKET_NAME = 'hate-speech'
ZIP_FILE_NAME = 'archive.zip'
LABEL = 'label'
TWEET = 'tweet'
MODEL_NAME = 'model.h5'
APP_HOST = "0.0.0.0"
APP_PORT = 8080

# Data ingestion constants
#dst_folder_path = "/path/to/destination/folder/"
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
#DATA_INGESTION_IMBALANCE_DATA_DIR = "imbalanced_data.csv"
#DATA_INGESTION_RAW_DATA_DIR = "raw_data.csv"
DATASET_DIR =  "labeled_data.csv"