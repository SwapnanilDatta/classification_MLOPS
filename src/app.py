import streamlit as st
import pandas as pd
import joblib

# Load saved components
model = joblib.load("models/rf_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
column_transformer = joblib.load("models/column_transformer.pkl")
churn_encoder = joblib.load("models/churn_encoder.pkl")

# Define column groups
binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
categorical_columns = ['InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaymentMethod', 'MultipleLines']
numeric_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Streamlit UI
st.title("📞 Telco Customer Churn Prediction")
st.markdown("Enter customer details to predict churn status.")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_back = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movie = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total = st.number_input("Total Charges", min_value=0.0, value=50.0)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    try:
        # Construct input DataFrame
        raw_input = pd.DataFrame([{
            'gender': gender,
            'Partner': partner,
            'Dependents': dependents,
            'PhoneService': phone,
            'PaperlessBilling': paperless,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_back,
            'DeviceProtection': device,
            'TechSupport': tech,
            'StreamingTV': tv,
            'StreamingMovies': movie,
            'Contract': contract,
            'PaymentMethod': payment,
            'MultipleLines': lines,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'tenure': int(tenure),
            'MonthlyCharges': float(monthly),
            'TotalCharges': float(total)
        }])

        # Apply Label Encoding to binary columns
        for col in binary_columns:
            le = label_encoders[col]
            raw_input[col] = raw_input[col].astype(str)  # ensure str
            raw_input[col] = le.transform(raw_input[col])

        # Transform with ColumnTransformer
        transformed_input = column_transformer.transform(raw_input)
        transformed_df = pd.DataFrame(transformed_input)

        # Make prediction
        # Make prediction
        prediction = model.predict(transformed_df)[0]
        proba = model.predict_proba(transformed_df)[0][1]

        # Safely decode prediction
        final = churn_encoder.inverse_transform([int(prediction)])[0]


        # Display result
        st.subheader("🔍 Prediction Result")
        if final == "Yes":
            st.error(f"⚠️ This customer is likely to **CHURN**. Confidence: {proba:.2%}")
        else:
            st.success(f"✅ This customer is likely to **STAY**. Confidence: {(1 - proba):.2%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
