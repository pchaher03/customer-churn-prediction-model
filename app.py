import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
import shap
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    return joblib.load("churn_models.joblib")

artifacts = load_artifacts()

# Sidebar Setup
st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest"])

active_model = artifacts["logistic_regression"] if model_choice == "Logistic Regression" else artifacts["random_forest"]

# Functions

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
    
    return df.reindex(columns=feature_columns, fill_value=0)

def plot_risk_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#d4edda"},
                {'range': [30, 60], 'color': "#fff3cd"},
                {'range': [60, 100], 'color': "#f8d7da"}
            ]
        }
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=20))
    return fig

# Main Page Setup

st.title("Customer Churn Risk & Retention Portal")

st.subheader("Customer Database Record")
st.caption("Double-click any cell below to update customer feature values in real time.")

default_customer_record = pd.DataFrame([{
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 12,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 65.0,
    'TotalCharges': 780.0
}])

raw_df = st.data_editor(
    default_customer_record,
    hide_index=True,
    num_rows="fixed",
    column_config={
        "gender": st.column_config.SelectboxColumn("Gender", options=["Female", "Male"], required=True),
        "SeniorCitizen": st.column_config.SelectboxColumn("Senior Citizen", options=[0, 1], required=True),
        "Partner": st.column_config.SelectboxColumn("Partner", options=["Yes", "No"], required=True),
        "Dependents": st.column_config.SelectboxColumn("Dependents", options=["Yes", "No"], required=True),
        "tenure": st.column_config.NumberColumn("Tenure (Months)", min_value=0, max_value=72, step=1, required=True),
        "PhoneService": st.column_config.SelectboxColumn("Phone Service", options=["Yes", "No"], required=True),
        "MultipleLines": st.column_config.SelectboxColumn("Multiple Lines", options=["No", "Yes", "No phone service"], required=True),
        "InternetService": st.column_config.SelectboxColumn("Internet Service", options=["DSL", "Fiber optic", "No"], required=True),
        "OnlineSecurity": st.column_config.SelectboxColumn("Online Security", options=["No", "Yes", "No internet service"], required=True),
        "OnlineBackup": st.column_config.SelectboxColumn("Online Backup", options=["Yes", "No", "No internet service"], required=True),
        "DeviceProtection": st.column_config.SelectboxColumn("Device Protection", options=["No", "Yes", "No internet service"], required=True),
        "TechSupport": st.column_config.SelectboxColumn("Tech Support", options=["No", "Yes", "No internet service"], required=True),
        "StreamingTV": st.column_config.SelectboxColumn("Streaming TV", options=["No", "Yes", "No internet service"], required=True),
        "StreamingMovies": st.column_config.SelectboxColumn("Streaming Movies", options=["No", "Yes", "No internet service"], required=True),
        "Contract": st.column_config.SelectboxColumn("Contract", options=["Month-to-month", "One year", "Two year"], required=True),
        "PaperlessBilling": st.column_config.SelectboxColumn("Paperless Billing", options=["Yes", "No"], required=True),
        "PaymentMethod": st.column_config.SelectboxColumn("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], required=True),
        "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges ($)", format="$%.2f", required=True),
        "TotalCharges": st.column_config.NumberColumn("Total Charges ($)", format="$%.2f", required=True),
    }
)

# Real-time prediction
X_single = preprocess_input(raw_df, artifacts["feature_columns"])
prob = active_model.predict_proba(X_single)[0][1]

# -------------------------------------------------------------------------
# DUAL-PATH SHAP CALCULATION (LOGISTIC REGRESSION VS RANDOM FOREST)
# -------------------------------------------------------------------------

if model_choice == "Logistic Regression":
    # PATH 1: Code 1 Logic (Uses saved artifacts directly)
    active_explainer = artifacts["explainer_lr"]
    
    try:
        explanation = active_explainer(X_single)
    except Exception:
        fallback_explainer = shap.Explainer(active_model.predict_proba, X_single)
        explanation = fallback_explainer(X_single)

    if len(explanation.shape) == 3:
        single_shap = explanation[0, :, 1]
        exp_slice = explanation[0, :, 1]
    elif len(explanation.shape) == 2:
        single_shap = explanation[0, :]
        exp_slice = explanation[0]
    else:
        single_shap = explanation
        exp_slice = explanation

    shap_vals_1d = single_shap.values
    is_explanation_obj = True

else:
    # PATH 2: Code 2 Logic (Uses dynamic TreeExplainer array extraction)
    explainer = shap.TreeExplainer(active_model)
    shap_vals = explainer.shap_values(X_single)

    if isinstance(shap_vals, list):
        shap_vals_1d = shap_vals[1][0]
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    elif len(shap_vals.shape) == 3:
        shap_vals_1d = shap_vals[0, :, 1]
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        shap_vals_1d = shap_vals[0]
        base_val = explainer.expected_value
    
    is_explanation_obj = False

# Build DataFrame for Bar Chart & Drivers
shap_df = pd.DataFrame({
    'Feature': X_single.columns,
    'SHAP Value': shap_vals_1d
}).sort_values(by='SHAP Value', key=abs, ascending=False)

# -------------------------------------------------------------------------
# DASHBOARD RESULTS VIEW
# -------------------------------------------------------------------------
st.markdown("---")
st.header("Customer Churn Risk Profile")

left_col, right_col = st.columns([1, 1.3])

with left_col:
    st.subheader("Churn Risk Score")
    fig_gauge = plot_risk_gauge(prob)
    st.plotly_chart(fig_gauge, use_container_width=True)

    if prob >= 0.60:
        st.error(f"🔴 **HIGH RISK** ({prob:.1%}): Immediate retention intervention required.")
    elif prob >= 0.30:
        st.warning(f"🟡 **MEDIUM RISK** ({prob:.1%}): Customer displays elevated churn indicators.")
    else:
        st.success(f"🟢 **LOW RISK** ({prob:.1%}): Customer profile is currently stable.")

with right_col:
    st.subheader("SHAP Feature Drivers")
    
    # Render Force Plot using the corresponding successful renderer for each model
    if is_explanation_obj:
        st_shap(shap.plots.force(exp_slice, plot_cmap=["#F08080", "#90EE90"]))
    else:
        st_shap(shap.plots.force(base_val, shap_vals_1d, X_single, plot_cmap=["#F08080", "#90EE90"]))

    top_risk_driver = shap_df[shap_df['SHAP Value'] > 0].head(1)
    top_retention_driver = shap_df[shap_df['SHAP Value'] < 0].head(1)

    summary_text_risk = ""
    if not top_risk_driver.empty:
        feat = top_risk_driver.iloc[0]['Feature']
        summary_text_risk += f"🔴 **Primary Risk Driver:** `{feat}` is pushing churn probability **UP**.\n\n"

    summary_text_retention = ""
    if not top_retention_driver.empty:
        feat = top_retention_driver.iloc[0]['Feature']
        summary_text_retention += f"🟢 **Primary Retention Driver:** `{feat}` is keeping churn probability **DOWN**."

    st.info(summary_text_risk)
    st.info(summary_text_retention)

with st.expander("Detailed Feature Contribution Breakdown (Bar Chart)"):
    colors = ['lightcoral' if val >= 0 else 'lightgreen' for val in shap_df['SHAP Value']]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
    ax.set_xlabel("SHAP Value")
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.invert_yaxis()
    st.pyplot(fig)