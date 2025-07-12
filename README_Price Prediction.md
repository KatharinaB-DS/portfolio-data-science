Author: Katarzyna Brzeski.
4th-year Informatics student, AI & Data Science specialization

# Real Estate Price Prediction

A machine learning project focused on predicting housing prices in New York City using advanced regression models and feature engineering techniques.

## Project Overview

This project aims to build a robust regression model to predict real estate prices based on a variety of features including location, size, number of rooms, and property type. The dataset comes from the NYC property market and contains detailed property-level information.

##  Key Steps

- **Data Cleaning**: Removed outliers, handled missing values, standardized formats.
- **Feature Engineering**:
  - `LUXURY_HOME`: Multiconditional flag for high-end properties.
  - `PRICE_PER_SQFT` & `LOG_PRICE_PER_SQFT`: Price per unit area (raw & log-transformed).
  - `ROOMS_PER_SQFT`: Indicator of space comfort.
  - `LOCATION_CATEGORY`: Manual clustering of geographic coordinates.
- **Exploratory Data Analysis**: Visualizations to assess correlations and distribution patterns.
- **Modeling**:
  - Baseline model (mean price)
  - Linear Regression
  - Random Forest Regressor (+ hyperparameter tuning with GridSearchCV)
  - XGBoost Regressor (+ optimized)
- **Evaluation Metrics**:
  - RMSE (Root Mean Squared Error)
  - R² Score (Explained Variance)
- **Deployment**:
  - Flask-based API accepting property features and returning predicted log-price.
  - Tested with real-world examples.

## Model Performance


Model  
RMSE
R²
Remarks
Mean (baseline)
0.8311
-0.0000
No modeling, just the mean 
Linear Regression
0.2990
0.8705
Simple model, but performed well
Random Forest Regressor
0.0403
0.9976
Very precise model
Random Forest (tuned) 
0.0398 


0.9977


Minimal improvement
XGBoost
0.0423
0.9974
Very good result 
XGBoost (tuned)
0.0335
0.9984
Best prediction quality



## API Demo

The model is deployed locally via a Flask API.

- **POST** `/predict`
- Accepts JSON input with all required features.
- Returns predicted `LOG_PRICE`.

```json
{
  "BEDS": 3,
  "BATH": 2,
  "PROPERTYSQFT": 1000,
  "LATITUDE": 40.7306,
  "LONGITUDE": -73.9352,
  "PRICE_PER_SQFT": 1100,
  "LUXURY_HOME": 1,
  "BOROUGH_Manhattan": 1,
  "TYPE_Condo for sale": 1,
  "LOCATION_CATEGORY_NorthEast": 1
}

Notebooks & Files
eda_modeling.ipynb — full analysis and model training

app.py — Flask API for deployment

model.pkl — serialized XGBoost model

feature_columns.pkl — expected features for inference

Real Estate Price Prediction.pdf — complete report (in English)

Technologies Used
Python (pandas, matplotlib, seaborn, scikit-learn, xgboost)

Jupyter Notebook

Flask (for API)

GridSearchCV (for tuning)

One-Hot Encoding for categorical features

Final Thoughts
The XGBoost model, especially after tuning, provides highly accurate predictions and captures the complexity of the NYC housing market. The inclusion of domain-informed features like LUXURY_HOME significantly boosted model performance. The deployed API makes it easy to integrate this solution into real-world applications.
