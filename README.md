# Fraud Detection in Financial Transactions
## Project Overview

This project focuses on detecting fraudulent financial transactions using machine learning. The objective is to identify high-risk transactions by analyzing transaction behavior, user activity, and contextual features, while handling strong class imbalance and ensuring model interpretability. The final solution delivers a robust, deployment-ready fraud detection model supported by explainable insights.

## Problem Statement
Financial fraud is a rare but high-impact event. In this dataset, fraudulent transactions represent approximately 5% of all records, making traditional accuracy-based modeling ineffective. The goal of this project is to:

- Accurately detect fraudulent transactions
- Minimize missed fraud cases (high recall)
- Control false positives
- Provide interpretable insights for business decision-making

## Dataset Description
The dataset contains 50,000 financial transactions with the following attributes:
- Transaction amount and time
- Payment method
- Device used
- Location
- User transaction history and velocity indicators
- Fraud label (Fraudulent / Non-Fraudulent)
The dataset includes missing values, duplicate records, mixed data types, and significant class imbalance.

## Project Workflow
### 1. Data Cleaning & Preprocessing
- Removed duplicate transactions
- Fixed incorrect data types
- Handled missing values using statistical and categorical imputation
- Encoded categorical variables using one-hot encoding
- Scaled numeric features using StandardScaler

### 2. Exploratory Data Analysis (EDA)
- Analyzed transaction amount distributions and outliers
- Examined fraud patterns by:
   - Payment method
   - Device type
   - Location
   - Time of transaction
   - Transaction velocity (last 24 hours)
- Visualized fraud trends using bar plots, box plots, heatmaps, and KDE plots

### 3. Feature Engineering & Selection
- Engineered behavioral features such as:
   - Transaction velocity
   - Recent activity indicators
- Removed zero-variance features
- Applied Mutual Information to select the most predictive features
- Reduced dimensionality while preserving predictive power

### 4. Handling Class Imbalance
- Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset
- Maintained a controlled fraud ratio to avoid overfitting
- Used stratified train-test splitting to preserve class distribution

### 5. Modeling & Evaluation
Trained and evaluated the following classification models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
- Evaluation Metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
   - Confusion Matrix
   - ROC Curves

Threshold tuning was applied (0.2) to improve fraud recall, recognizing the higher cost of missed fraud.

### 6. Best Model Selection
XGBoost was selected as the final model due to:
- Highest ROC-AUC and F1-score
- Strong recall for fraudulent transactions
- Robust performance on imbalanced data
- Ability to capture complex, non-linear patterns

### 7. Model Explainability
  Generated feature importance using:
  - Gain
  - Weight
  - Cover
- Identified key fraud drivers such as transaction velocity, payment method, and device usage

## Key Insights
- Fraud is more strongly linked to behavioral patterns than transaction amount alone
- High transaction frequency within short time windows is a strong fraud signal
- Certain payment methods and devices exhibit higher fraud risk
- Threshold tuning significantly improves real-world fraud detection performance

## Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- Imbalanced-learn (SMOTE)
- XGBoost
- Jupyter Notebook

## How to Run the Project
- Clone the repository
- Install required dependencies
- Open the Jupyter Notebook
- Run cells sequentially to reproduce results

## Future Improvements
- Time-series modeling with transaction sequences
- Real-time fraud detection pipeline
- Cost-sensitive learning and profit-based evaluation
- Deployment as an API or dashboard

## Author
Hiren Bista
Graduate Student – Geography (GIS & Data Analytics)
University of North Texas
