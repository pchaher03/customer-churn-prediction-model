# Telco Customer Churn Prediction & Deployment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end Machine Learning project predicting customer attrition for telecom services, complete with data preprocessing, model evaluation, pipeline serialization, and an interactive Streamlit web application.

## Live demo & web app

**Interactive app:** [Click here to launch the Streamlit App]()

## Business overview

Customer retention is critical for subscription-based telecommunications companies. Acquiring new customers can cost up to 5 times more than retaining existing ones. 

The goal of this project is to build a predictive machine learning pipeline that identifies high-risk customers before they churn, enabling marketing and customer success teams to take proactive retention measures.

## Project architecture & workflow

[ KaggleHub Data Ingestion ] ➔ [ EDA & Data Cleaning ] ➔ [ Feature Engineering & Scaling ] ➔ [ Model Training & Tuning ] ➔ [ SHAP Evaluation ] ➔ [ Artifact Serialization (.joblib) ] ➔ [ Streamlit App (app.py) ]

## Repository structure

```
customer-churn-prediction-model/

    ├── .gitignore                  
    ├── app.py                      # Streamlit Web Application
    ├── churn_models.joblib         # Serialized ML pipeline artifact
    ├── churn_predictor.ipynb       # Main Jupyter Notebook (EDA, Modeling, Analysis)
    ├── README.md                  
    ├── requirements.txt            # Python dependencies
    ├── .dockerignore               
    ├── docker-compose.yml          
    └── Dockerfile                  
```

## Key results & model performance

We evaluated a baseline **Logistic Regression** model against an ensemble **Random Forest Classifier** on the positive churn class (Class 1).

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.74** | 0.51 | **0.78** | **0.61** | **0.8415** |
| **Random Forest** | **0.79** | **0.63** | 0.48 | 0.54 | 0.8280 |

### Model selection insight
* **Logistic Regression outperforms for retention efforts:** while Random Forest achieves higher overall accuracy (79%), **Logistic Regression captures significantly more churners** with a **78% Recall** compared to Random Forest's **48%**.
* **Business trade-off:** in customer churn, missing a churning customer (false negative) carries a higher business cost than sending a retention offer to a loyal customer (false positive). 

### Key drivers & SHAP interpretability:
* **Contract type:** month-to-month contracts exhibit the strongest positive SHAP values toward churn risk.
* **Tenure:** shorter customer tenure (first 1–12 months) significantly increases churn probability.
* **Payment method:** electronic check payments correlate strongly with higher churn rates.

## How to run streamlit app locally

Follow these steps to run the Streamlit app locally on your machine:

### 1. Clone the repository

```bash
git clone https://github.com/pchaher03/customer-churn-prediction-model
cd customer-churn-prediction-model
```

### 2. Build docker image and run container

```
docker compose up --build
```

**Important note for Windows users**: if you encounter a file-locking issue with Docker Desktop's BuildKit engine, run $env:DOCKER_BUILDKIT=0 in PowerShell prior to building.

### 3. See app in: http://localhost:8501/
