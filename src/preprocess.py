import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
   
    try:
        df = pd.read_csv('data/Tele.csv')
        logger.info("Data loaded successfully with shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise

def handle_missing_values(df):
   
    df = df.replace(' ', np.nan)
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill numeric columns with median
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
        
    logger.info("Missing values handled")
    return df

# ...existing code...

def encode_categorical_features(df):
    """Encode categorical features using Label Encoding and One-Hot Encoding"""
    
    # Columns for Label Encoding (binary categories)
    binary_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                     'PaperlessBilling', 'Churn']
    
    # Columns for One-Hot Encoding (multiple categories)
    categorical_columns = ['InternetService', 'OnlineSecurity', 'OnlineBackup',
                         'DeviceProtection', 'TechSupport', 'StreamingTV',
                         'StreamingMovies', 'Contract', 'PaymentMethod']
    
    # Apply Label Encoding to binary columns
    label_encoder = LabelEncoder()
    for column in binary_columns:
        df[column] = label_encoder.fit_transform(df[column])
        logger.info(f"Label encoded column: {column}")
    
    # Apply One-Hot Encoding to categorical columns
    ct = ColumnTransformer(
        [('encoder', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)],
        remainder='passthrough'
    )
    
    # Get feature names for one-hot encoded columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_features = encoder.fit(df[categorical_columns])
    feature_names = encoder.get_feature_names_out(categorical_columns)
    
    # Transform the data
    encoded_array = ct.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names)
    
    # Drop original categorical columns and concatenate with encoded features
    df = df.drop(columns=categorical_columns)
    df = pd.concat([df, encoded_df], axis=1)
    
    logger.info("One-hot encoding completed")
    return df

# ...existing code...

def preprocess_data():
    """Main preprocessing function"""
    try:
        # Load the data
        df = load_data()
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Encode categorical features
        df = encode_categorical_features(df)
        
        # Select only numeric columns for final output
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        logger.info("Preprocessing completed successfully. Final shape: %s", numeric_df.shape)
        return numeric_df
        
    except Exception as e:
        logger.error("Error in preprocessing: %s", e)
        raise

if __name__ == "__main__":
    processed_data = preprocess_data()
    # Save preprocessed data
    processed_data.to_csv('data/processed_telco.csv', index=False)
    logger.info("Preprocessed data saved to data/processed_telco.csv")