import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    return joblib.load("churn_models.joblib")

artifacts = load_artifacts()

# Sidebar Setup
st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
active_model = artifacts["logistic_regression"] if model_choice == "Logistic Regression" else artifacts["random_forest"]

def preprocess_input(raw_df, feature_columns):
    df = raw_df.copy()
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    if 'Churn' in df.columns:
        df.drop(columns=['Churn'], inplace=True)
        
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)).fillna(0.0)
        
    no_service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in no_service_cols:
        if col in df.columns:
            df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})
            
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col] == 'Yes').astype(int)
            
    if 'gender' in df.columns:
        df['gender'] = (df['gender'] == 'Female').astype(int)
        
    cat_cols = ['InternetService', 'Contract', 'PaymentMethod'] + [c for c in no_service_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    
    # Align columns to match training set (adds missing dummies as 0)
    return df.reindex(columns=feature_columns, fill_value=0)

st.title("Customer Churn Risk & Retention Portal")

tab1, tab2 = st.tabs(["Single Customer Risk", "Batch Prediction"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", value=65.0)
        total_charges = st.number_input("Total Charges ($)", value=780.0)

    if st.button("Calculate Churn Probability", type="primary"):
        raw_df = pd.DataFrame([{
            'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }])
        
        X_single = preprocess_input(raw_df, artifacts["feature_columns"])
        prob = active_model.predict_proba(X_single)[0][1]
        
        st.metric("Churn Probability", f"{prob:.1%}")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV File (Raw or Cleaned)", type=["csv"])
    if uploaded_file:
        raw_batch = pd.read_csv(uploaded_file)
        st.write("Preview:", raw_batch.head(3))
        
        if st.button("Run Batch Prediction"):
            # Works whether CSV is raw or already cleaned
            X_batch = preprocess_input(raw_batch, artifacts["feature_columns"])
            raw_batch['Churn_Probability'] = active_model.predict_proba(X_batch)[:, 1]
            st.dataframe(raw_batch)