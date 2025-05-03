# CreditRisk_FeatureEng

## Overview

This project develops a credit risk scoring model using a logistic regression model. The goal is to predict the probability of default (PD) for loan applicants and assign credit scores to clients based on a scorecard system. The model processes the data through exploratory data analysis (EDA), feature engineering, Information Value, WoE (Weight of Evidence) transformation, and logistic regression modeling, culminating in a scorecard that maps client features to credit scores.

## Dataset

- **File**: `credit_dataset.csv`

- **Description**: Contains loan application data with various features and the target variable `loan_status` (used to derive `PD`)
  
- **Target Variable**: `PD` (Probability of Default), derived from `loan_status`:

  - PD = 0: Fully Paid
  - PD = 1: Default, In Grace Period, Late (16-30 days), Late (31-120 days)
 
- **Dataset Size Note**: The original `credit_dataset.csv` file is 229 MB, which exceeds GitHub's file size limit for uploads (100 MB). To address this, the dataset was cleaned and reduced in size by:
  - Filtering relevant `loan_status` categories.
  - Dropping columns with high missing values (>80%) and zero variance.
  - Removing non-predictive columns.
  - The reduced dataset, `credit_test_data.csv`, was then used for the project and uploaded to GitHub.


## Steps
1. **Data Preprocessing**:
   - Filtered the dataset to include only relevant `loan_status` categories (`Fully Paid`, `Default`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)`).
   - Dropped columns with over 80% missing values and columns with zero variance.
   - Handled outliers by clipping numeric features using the interquartile range (IQR) method.
   - Binned numeric features into 5 quantiles using `pd.qcut` to create categorical variables.

2. **Feature Engineering**:
   - Converted all features to strings to treat them as categorical variables.
   - Calculated Weight of Evidence (WoE) and Information Value (IV) for each feature using a custom `woe_iv` function.
   - Selected features with IV between 0.02 and 0.5 for modeling.

3. **Model Training**:
   - Split the data into training and test sets (80/20) with stratification to ensure both `PD=0` and `PD=1` are represented.
   - Transformed features into WoE values using the computed `woe_tables`.
   - Trained a logistic regression model with balanced class weights to handle the imbalanced target variable.

4. **Scorecard Creation**:
   - Defined scaling parameters: `base_score = 700` (at odds 20:1), `PDO = 20`, `factor = PDO / ln(2)`, `offset = base_score - factor * ln(score_odds)`.
   - Generated a scorecard by mapping WoE values to points using the formula `Points = -factor * coefficient * WoE`.
   - Included an intercept term in the scorecard.

5. **Client Scoring**:
   - Calculated credit scores for the first 10 clients in the test set by matching their feature values to the scorecard categories and summing the points, starting from the `offset`.
   - Created a DataFrame with `id` and `score` columns for these clients.

## Requirements
- Python 3.12
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `category_encoders`

Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders
```

## Challenges and Solutions

- **WoE Calculation Failure**: The WOEEncoder library failed to generate WoE mappings due to data issues. We resolved this by manually computing WoE values using a custom woe_iv function.
- **Imbalanced Data**: The target variable PD was imbalanced, causing issues with WoE calculation. We used stratification in the train-test split and skipped categories with zero events or non-events.
- **Binning Issues**: Test data binning failed due to pre-binned interval objects. We removed the re-binning step since the data was already binned during preprocessing.

## Authors

- Ana Luisa Espinoza López
- Daniel Sánchez López

- Date: May 02, 2025
