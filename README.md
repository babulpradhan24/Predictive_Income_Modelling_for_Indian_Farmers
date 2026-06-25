# Predictive Income Modelling for Indian Farmers

## Overview

Developed an end-to-end machine learning pipeline to predict the annual income of Indian farmers using demographic, agricultural, climatic, and socioeconomic attributes. The project was completed as part of the **L&T Finance Challenge (Trilytics'25)**, where our team **Gravity** secured **2nd Rank in the Preliminary Round**.

---

## Problem Statement

Accurate estimation of farmer income is essential for credit assessment, loan disbursement, and financial inclusion. The objective was to build a regression model capable of predicting farmer income from structured tabular data while minimizing Mean Absolute Percentage Error (MAPE).

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost
* CatBoost
* SHAP
* AutoFeat
* Optuna
* Matplotlib

---

## Project Pipeline

```
Raw Dataset
      │
      ▼
Data Cleaning & Preprocessing
      │
      ▼
Encoding & Feature Scaling
      │
      ▼
Exploratory Data Analysis
      │
      ▼
AutoFeat Feature Engineering
      │
      ▼
Model Training
(Random Forest, XGBoost,
LightGBM, CatBoost, etc.)
      │
      ▼
Hyperparameter Optimization
(Optuna)
      │
      ▼
SHAP Feature Selection
      │
      ▼
Final Model
```

---

## Key Features

* Cleaned and preprocessed structured agricultural data with extensive missing-value handling.
* Applied Label Encoding and Target Encoding for categorical variables.
* Performed feature engineering using AutoFeat to capture non-linear interactions.
* Compared multiple regression algorithms including:

  * LightGBM
  * XGBoost
  * CatBoost
  * Random Forest
  * Extra Trees
  * HistGradientBoosting
  * Linear Regression
  * Ridge
  * Lasso
  * ElasticNet
  * TabNet
* Optimized model parameters using Optuna.
* Reduced feature space through SHAP-based feature importance analysis.

---

## Results

* Generated 30+ engineered features using AutoFeat.
* Reduced MAPE from **0.2130** (baseline) to **0.2014** after SHAP-based feature pruning.
* Evaluated 10+ machine learning models before selecting the final ensemble pipeline.
* Secured **2nd Rank in the Preliminary Round** of the L&T Finance Challenge (Trilytics'25).

---

## Repository Structure

```
├── Farmer_Income_Prediction.ipynb
├── Presentation.pdf
├── README.md
└── requirements.txt
```

---

## Future Improvements

* Integrate satellite imagery and NDVI data.
* Incorporate weather forecasts and government scheme data.
* Experiment with advanced ensemble and stacking techniques.
* Deploy the trained model as a web application.

---
