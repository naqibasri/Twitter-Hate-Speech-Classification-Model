import sys
import os
import shutil
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import ModelPusherArtifacts, ModelEvaluationArtifacts

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_evaluation_artifacts: ModelEvaluationArtifacts):
        """
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_pusher_config = model_pusher_config
        self.model_evaluation_artifacts = model_evaluation_artifacts

    def copy_file_locally(self) -> None:
        shutil.copy(self.model_pusher_config.TRAINED_MODEL_PATH, self.model_pusher_config.BEST_MODEL_PATH)
    
    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
            Method Name :   initiate_model_pusher
            Description :   This method initiates model pusher.

            Output      :    Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")
        try:
            # Uploading the model to gcloud storage
            os.makedirs(self.model_pusher_config.BEST_MODEL_PATH, exist_ok=True)
            
            self.copy_file_locally()
            logging.info("Uploaded best model to best_Model_saved folder")

            # Saving the model pusher artifacts
            model_pusher_artifact = ModelPusherArtifacts(
                best_model_path=self.model_pusher_config.BEST_MODEL_PATH
            )
            logging.info("Exited the initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
