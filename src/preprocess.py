import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def load_data():
    try:
        df = pd.read_csv('data/Tele.csv')
        logger.info(f"✅ Data loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

def handle_missing_values(df):
    df = df.replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    logger.info("✅ Missing values handled")
    return df

def encode_features(df):
    binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    categorical_columns = ['InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV',
                           'StreamingMovies', 'Contract', 'PaymentMethod', 'MultipleLines']

    label_encoders = {}

    # Label encode binary columns (excluding SeniorCitizen)
    for col in binary_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        logger.info(f"Label encoded: {col}")

    # Convert SeniorCitizen from Yes/No → 1/0 if needed (already numeric in some versions)
    if df['SeniorCitizen'].dtype == 'object':
        df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})

    # ColumnTransformer for categorical columns (OneHotEncoding)
    ct = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
        ],
        remainder='passthrough'
    )

    input_features = df.drop(columns=['Churn'])
    ct.fit(input_features)
    encoded_array = ct.transform(input_features)
    encoded_df = pd.DataFrame(encoded_array)

    logger.info("✅ One-hot encoding completed")

    # Encode Churn separately
    churn_le = LabelEncoder()
    df['Churn'] = churn_le.fit_transform(df['Churn'])

    # Save the encoders and transformer
    os.makedirs("models", exist_ok=True)
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(ct, "models/column_transformer.pkl")
    joblib.dump(churn_le, "models/churn_encoder.pkl")
    logger.info("✅ Encoders and transformer saved")

    # Concatenate encoded inputs with Churn
    encoded_df['Churn'] = df['Churn'].values

    return encoded_df

def preprocess_data():
    try:
        df = load_data()
        df = handle_missing_values(df)
        processed_df = encode_features(df)

        os.makedirs("data", exist_ok=True)
        processed_df.to_csv("data/processed_telco.csv", index=False)
        logger.info("✅ Final preprocessed data saved to data/processed_telco.csv")

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()
