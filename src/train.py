import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import logging
import joblib
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

mlflow.set_tracking_uri("https://dagshub.com/dattaswapnanil/classification_MLOPS.mlflow")
mlflow.set_experiment("telco-churn-prediction")

def load_data(path='data/processed_telco.csv'):
    try:
        df = pd.read_csv(path)
        logger.info("Loaded preprocessed data with shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise

def train_model(X_train, y_train):
    # Define model and parameter grid
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

   
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logger.info("Best parameters: %s", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metrics

def main():
    try:
        df = load_data()

       
        X = df.drop(columns='Churn')
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with mlflow.start_run():
     
            best_model = train_model(X_train, y_train)

            # Evaluate the model
            metrics = evaluate_model(best_model, X_test, y_test)
            logger.info("Model Evaluation: %s", metrics)

            # Log only parameters and metrics
            mlflow.log_params(best_model.get_params())
            mlflow.log_metrics(metrics)

            # Save model locally (skip MLflow log_model to avoid DagsHub error)
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_model, "models/rf_model.pkl")
            logger.info("Saved model to models/rf_model.pkl")


    except Exception as e:
        logger.error("Training pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()
