import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import logging
from pathlib import Path
import os
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI and authentication
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

mlflow.set_tracking_uri("https://dagshub.com/dattaswapnanil/classification_MLOPS.mlflow")
mlflow.set_experiment("telco-churn-prediction")

def load_processed_data():
    try:
        data = pd.read_csv('data/processed_telco.csv')
        logger.info("Processed data loaded successfully with shape: %s", data.shape)
        return data
    except Exception as e:
        logger.error("Error loading processed data: %s", e)
        raise

def prepare_training_data(df):
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info("Data split into training and test sets")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error("Error preparing training data: %s", e)
        raise

def train_model_with_grid_search(X_train, X_test, y_train, y_test):
    try:
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5]
        }

        rf = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )

        with mlflow.start_run():
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log best parameters
            mlflow.log_params(grid_search.best_params_)

            # Log all tried parameters as separate files
            for i, params in enumerate(grid_search.cv_results_["params"]):
                param_file = f"params/params_{i}.json"
                Path("params").mkdir(exist_ok=True)
                with open(param_file, "w") as f:
                    import json
                    json.dump(params, f)
                mlflow.log_artifact(param_file)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Save model manually using joblib
            Path("models").mkdir(exist_ok=True)
            local_model_path = "models/random_forest_model.pkl"
            joblib.dump(best_model, local_model_path)

            # Log model as artifact
            mlflow.log_artifact(local_model_path, artifact_path="model")

            logger.info("Model training completed with best hyperparameters")
            logger.info(f"Best Params: {grid_search.best_params_}")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")

            return best_model

    except Exception as e:
        logger.error("Error in model training: %s", e)
        raise

def save_model(model, path="models"):
    try:
        Path(path).mkdir(exist_ok=True)
        model_path = f"{path}/random_forest_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise

def main():
    try:
        data = load_processed_data()
        X_train, X_test, y_train, y_test = prepare_training_data(data)
        model = train_model_with_grid_search(X_train, X_test, y_train, y_test)
        save_model(model)
    except Exception as e:
        logger.error("Error in training pipeline: %s", e)
        raise

if __name__ == "__main__":
    main()
