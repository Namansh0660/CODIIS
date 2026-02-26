# MediRisk — Diabetic Patient Readmission Prediction

A clinical decision support system that predicts hospital readmission risk for diabetic patients using an ensemble of 8 machine learning models, with an interactive Streamlit dashboard and automated email alert workflow.

---

## Overview

Built on the [UCI Diabetic Readmission dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) (101,766 records, 50 features), this project covers the full ML pipeline — from EDA and preprocessing through model benchmarking and deployment — with a production-style dashboard for clinical use.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | pandas, NumPy, Matplotlib, Seaborn |
| ML Models | scikit-learn, XGBoost |
| Serialization | joblib |
| Dashboard | Streamlit |
| Alerts | smtplib (Gmail SMTP) |

---

## Project Structure

```
medirisk/
├── app.py                              # Streamlit dashboard
├── Healthcare.ipynb                    # Full ML pipeline notebook
├── diabetic_data.csv                   # Raw dataset (UCI)
├── model_predictions.xlsx              # Test-set predictions from all 8 models
├── readmission_model_reloaded.joblib   # Saved RF predictions array
└── README.md
```

---

## ML Pipeline

**Preprocessing**
- Replace `?` with `NaN`
- Binarize target: `readmitted != "NO"` → `1`, else `0`
- Label encode all categorical features
- `SimpleImputer(strategy="mean")` for missing values
- `train_test_split(test_size=0.2, random_state=42)`

**Models Trained**

| Model | Accuracy | AUC |
|---|---|---|
| KNN | 0.5461 | 0.5528 |
| Decision Tree | 0.5833 | 0.5814 |
| Naive Bayes | 0.5894 | 0.6452 |
| Logistic Regression | 0.6272 | 0.6696 |
| Voting Classifier | 0.6620 | 0.7195 |
| Random Forest | 0.6611 | 0.7182 |
| GBM | 0.6632 | 0.7218 |
| **XGBoost** | **0.6685** | **0.7304** |

**Output Files**
- `model_predictions.xlsx` — `True_Label` + predictions from all 8 models on the test set
- `readmission_model_reloaded.joblib` — Random Forest predictions array (mirrors `joblib.dump(rf_pred, ...)`)

---

## Dashboard Pages

**Overview & EDA** — dataset statistics, feature distributions, readmission breakdowns by age and time in hospital, correlation heatmap

**Model Performance** — visual score cards with accuracy/AUC fill-bars, confusion matrix per model, per-metric bar charts, model agreement matrix

**Prediction Explorer** — filter test-set predictions by true label or correctness, inspect any single row with per-model vote breakdown and consensus gauge

**Email Alert Centre** — loads RF predictions from `.joblib`, shows high-risk patient breakdown (true positives vs false positives), sends personalised care-alert emails via Gmail SMTP

---

## Setup

**1. Install dependencies**
```bash
pip install streamlit pandas openpyxl scikit-learn xgboost seaborn matplotlib joblib
```

**2. Place files in the same folder**
```
app.py
diabetic_data.csv
model_predictions.xlsx
readmission_model_reloaded.joblib
```

**3. Run**
```bash
streamlit run app.py
```

**4. Upload** `diabetic_data.csv` via the sidebar — the xlsx and joblib are loaded automatically from disk.

---

## Email Alerts

For the email alert workflow, the sender needs a **Gmail App Password** (not their regular password).

Generate one at: `Google Account → Security → 2-Step Verification → App Passwords`

High-risk patients (RF prediction = 1) receive an automated follow-up care notification. Bulk sending is supported if the dataset includes `name` and `email` columns.

---

## Internship Context

> Developed a diabetic patient readmission prediction system on 101,766 clinical records, benchmarking 8 ML models including XGBoost, GBM, Random Forest, and a Voting Classifier — achieving a best AUC of 0.73. Built an end-to-end pipeline covering preprocessing, EDA, K-Means patient clustering, and a Streamlit dashboard with automated email alerts for high-risk patient follow-up.

---

## Dataset

**UCI Diabetic 130-US Hospitals Dataset (1999–2008)**  
Strack, B. et al. — Impact of HbA1c Measurement on Hospital Readmission Rates. *BioMed Research International*, 2014.