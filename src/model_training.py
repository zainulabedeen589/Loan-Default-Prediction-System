import os
import pandas as pd
import sys
import joblib
import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
from src.logger import get_logger
from src.custom_exception import CustomException
from src.paths_config import *

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, data_path, params_path, model_save_path, experiment_name="Airline_Default_Prediction"):
        self.data_path = data_path
        self.params_path = params_path
        self.model_save_path = model_save_path
        self.experiment_name = experiment_name
        
        self.best_model = None
        self.metrics = None

        # Set tracking URI to ensure MLflow UI finds the data in the project root
        # This prevents the "blank UI" issue
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tracking_uri = f"file://{os.path.join(project_root, 'mlruns')}"
        mlflow.set_tracking_uri(self.tracking_uri)

    def load_data(self):
        try:
            logger.info("Loading data for model training...")
            data = pd.read_csv(self.data_path)
            logger.info("Data loaded successfully.")
            return data
        except Exception as e:
            raise CustomException(f"Error while loading data: {str(e)}", sys)
        
    def split_data(self, data):
        try:
            logger.info("Splitting data into train and test sets...")
            X = data.drop(columns='Default')
            y = data['Default']
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logger.info("Data splitting completed.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(f"Error while splitting data: {str(e)}", sys)
        
    def train_model(self, X_train, y_train, params):
        try:
            logger.info("Starting GridSearchCV with LightGBM...")
            lgbm = lgb.LGBMClassifier()
            grid_search = GridSearchCV(lgbm, param_grid=params, cv=3, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            self.best_model = grid_search.best_estimator_
            logger.info("Model training completed.")
            return grid_search.best_params_
        except Exception as e:
            raise CustomException(f"Error while training model: {str(e)}", sys)
        
    def evaluate_model(self, X_test, y_test):
        try:
            logger.info("Evaluating model...")
            y_pred = self.best_model.predict(X_test)

            self.metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }

            logger.info(f"Evaluation metrics: {self.metrics}")
            return self.metrics
        except Exception as e:
            raise CustomException(f"Error while evaluating model: {str(e)}", sys)
    
    def save_model(self):
        try:
            logger.info(f"Saving model to {self.model_save_path}")
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            joblib.dump(self.best_model, self.model_save_path)
            logger.info("Model saved successfully.")
        except Exception as e:
            raise CustomException(f"Error while saving model: {str(e)}", sys)
        
    def run(self):
        try:
            mlflow.set_experiment(self.experiment_name)
            
            with mlflow.start_run():
                # 1. Load Data
                data = self.load_data()

                # 2. Split Data
                X_train, X_test, y_train, y_test = self.split_data(data)

                # 3. Load Params from JSON
                with open(self.params_path, "r") as f:
                    params = json.load(f)
                
                logger.info(f"Hyperparameters loaded: {params}")
                mlflow.log_params({f"grid_{k}": v for k, v in params.items()})

                # 4. Train
                best_params = self.train_model(X_train, y_train, params)
                mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

                # 5. Evaluate
                metrics = self.evaluate_model(X_test, y_test)
                
                # Log metrics (excluding list-based data like confusion matrix)
                for metric, value in metrics.items():
                    if metric != "confusion_matrix":
                        mlflow.log_metric(metric, value)

                # 6. Log Confusion Matrix as an Artifact (JSON)
                cm_path = "confusion_matrix.json"
                with open(cm_path, "w") as f:
                    json.dump({"confusion_matrix": metrics["confusion_matrix"]}, f)
                mlflow.log_artifact(cm_path)
                
                # 7. Save and Log Model
                self.save_model()
                mlflow.sklearn.log_model(self.best_model, "model")
                
                logger.info(f"Run successful. Tracking URI: {self.tracking_uri}")

        except Exception as e:
            logger.error(f"Run failed: {str(e)}")
            mlflow.end_run(status="FAILED")
            raise e

if __name__ == "__main__":
    # Ensure constants are imported from paths_config
    model_trainer = ModelTraining(
        data_path=ENGINEERED_DATA_PATH, 
        params_path=PARAMS_PATH,  
        model_save_path=MODEL_SAVE_PATH
    )
    model_trainer.run()