# 📞 Telco Customer Churn Prediction

A machine learning project that predicts customer churn for a telecommunications company using end-to-end **MLOps best practices**.

![MLOps Workflow](https://img.shields.io/badge/MLOps-DVC%20%7C%20MLflow%20%7C%20Dagshub-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Project Description

This project builds a churn prediction pipeline for a telecom company by integrating:
- 🔁 Data versioning with **DVC**
- 📈 Experiment tracking with **MLflow**
- 🧠 Model training using **Random Forest**
- 🌐 Web-based UI via **Streamlit**

The goal is to identify customers likely to churn, enabling the company to implement **proactive retention strategies**.

---

## 🚀 Key Features

- ✅ **Data Preprocessing Pipeline:** Handles missing values, encoding, and transformations  
- 🌲 **Machine Learning Model:** RandomForest classifier with hyperparameter tuning via `GridSearchCV`  
- 🔬 **MLflow Integration:** Logs experiments, metrics, and models  
- 📦 **DVC:** Manages datasets and model files  
- 🖥️ **Streamlit UI:** Allows user-friendly prediction on input data

---

## 🧠 Technology Stack

### 🔧 Python Libraries:
- `Scikit-learn` – ML algorithms & evaluation
- `Pandas`, `NumPy` – Data handling & transformation
- `MLflow` – Experiment tracking
- `DVC` – Data & model version control
- `Streamlit` – Web interface for predictions

---

## 📊 Features Used for Prediction

- **Demographics:** Gender, Senior Citizen  
- **Account Info:** Contract Type, Payment Method  
- **Services:** Phone, Internet, Add-ons  
- **Usage Metrics:** Tenure, Monthly Charges, Total Charges  

---

## ⚙️ How to Run Locally

### ✅ Install Dependencies
```bash
uv add -r requirements.txt
```

### 🔁 Pull DVC-tracked Data & Models
Make sure you have DVC installed and configured, then run:
```bash
dvc pull
```

### 🚀 Launch Streamlit App
To start the user interface for predictions, run:
```bash
streamlit run src/app.py
```

---

## 📈 MLflow Tracking UI

View experiment runs, metrics, parameters, and model artifacts here:  
🔗 [MLflow UI Link](https://dagshub.com/dattaswapnanil/classification_MLOPS/experiments#/experiment/m_43d36c7611c840c58c352fb2373db77b)

---

## ☁️ Dagshub Integration

All datasets and models are versioned and stored on Dagshub via DVC:  
🔗 [Dagshub Repo](https://dagshub.com/dattaswapnanil/classification_MLOPS)

---

## ✅ Current Results (Sample Metrics)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 81.5%  |
| Precision | 70%    |
| Recall    | 52.5%  |
| F1 Score  | 60%    |

---

## 🧑‍💻 Author

**Swapnanil Datta**  

📧 dattaswapnanil@gmail.com

---

## 📜 License

This project is licensed under the **MIT License**.
