# Telco Customer Churn Prediction & Deployment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end Machine Learning project predicting customer attrition for telecom services, complete with data preprocessing, model evaluation, pipeline serialization, and an interactive Streamlit web application.

---

## Live demo & web app

**Interactive app:** [Click here to launch the Streamlit App]()

---

## Business overview

Customer retention is critical for subscription-based telecommunications companies. Acquiring new customers can cost up to 5 times more than retaining existing ones. 

The goal of this project is to build a predictive machine learning pipeline that identifies high-risk customers before they churn, enabling marketing and customer success teams to take proactive retention measures.

---

## Project architecture & workflow

[ KaggleHub Data Ingestion ]<br/>&#8595;<br/>
[ EDA & Data Cleaning ] (Type Conversion, Missing Values, Outliers)<br/>&#8595;<br/>
[ Feature Engineering & Scaling ]<br/>&#8595;<br/>
[ Model Training & Tuning ]
(Logistic Regression vs. Random Forest)<br/>&#8595;<br/>
[ Artifact Serialization ] (.joblib)<br/>&#8595;<br/>
[ Interactive Streamlit App ] (app.py)

---

## Repository structure

```
customer-churn-prediction-model/

    ├── .gitignore                  
    ├── app.py                      # Streamlit Web Application
    ├── churn_models.joblib         # Serialized ML pipeline artifact
    ├── churn_predictor.ipynb       # Main Jupyter Notebook (EDA, Modeling, Analysis)
    ├── README.md                   # Project documentation
    └── requirements.txt            # Python dependencies
```

## Key results & model performance

Trained and evaluated a baseline **Logistic Regression** model against an ensemble **Random Forest Classifier** to assess precision, recall, and overall ROC-AUC performance on the positive churn class (Class 1).

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.74** | 0.51 | **0.78** | **0.61** | **0.8415** |
| **Random Forest** | **0.79** | **0.63** | 0.48 | 0.54 | 0.8280 |

### Model selection insight
* **Logistic Regression outperforms for retention efforts:** While Random Forest achieves a higher overall accuracy (79%), **Logistic Regression captures significantly more churners** with a **78% Recall** compared to Random Forest's **48%**.
* **Business trade-off:** In customer churn, the cost of missing a churning customer (false negative) is much higher than sending a retention offer to a loyal customer (false positive). Thus, Logistic Regression is the preferred model for this pipeline.

### Key drivers of churn:
* **Contract type:** Month-to-month contracts showed significantly higher churn rates compared to two-year contracts.
* **Tenure:** Newer customers (first 1–12 months) are at the highest risk of leaving.
* **Payment method:** Electronic check payments correlated with higher churn propensity.

---

## How to run locally

Follow these steps to run the Jupyter Notebook or Streamlit App locally on your machine:

### 1. Clone the repository

```bash
git clone https://github.com/pchaher03/customer-churn-prediction-model
cd customer-churn-prediction-model
```

### 2. Create a new environment with Python 3.10
*(Miniconda with conda-forge channel priority was used during development)*
```
# Configure conda-forge with strict priority

conda config --add channels conda-forge
conda config --set channel_priority strict

# Create and activate environment

conda create --name your-env python=3.10 -y
conda activate your-env
```

### 4. Install dependencies
```
conda install --file requirements.txt
```

### 5. Run the streamlit app *(If unfortunately the link from the beginning does not work)*
```
streamlit run app.py
```
