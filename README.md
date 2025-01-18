# Credit Risk Assessment

This project, developed on **Databricks**, predicts the likelihood of default payment in the next month using the **Default of Credit Card Clients Dataset** from the UCI Machine Learning Repository. It implements **Logistic Regression** and **XGBoost** for classification, with **SHAP (SHapley Additive exPlanations)** for model interpretability.

## Dataset
**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
**Authors**: I-Cheng Yeh (Department of Information Management, Chung Hua University, Taiwan)  
The dataset includes features such as credit limit, age, marital status, education, historical payment and bill amounts, and the target variable (`DEFAULT_PAYMENT_NEXT_MONTH`).

## Project Pipeline
1. **Data Preparation**:
   - Data loaded into a Spark DataFrame, cleaned, and normalized.
   - Columns renamed for interpretability (e.g., `X1` → `LIMIT_BAL`).
2. **Exploratory Data Analysis**:
   - Class distribution (default vs. non-default).
   - Correlation matrix and feature relationships (e.g., payment history vs. credit limit).
   - Feature distributions (e.g., credit limit, age).
3. **Model Development**:
   - **Logistic Regression**: Baseline binary classification model.
   - **XGBoost Classifier**: Gradient boosting model for advanced feature interactions.
4. **Model Interpretation with SHAP**:
   - SHAP summary plot highlights feature importance (e.g., `PAY_0`, bill amounts).
   - Explains individual predictions.

## Insights
- **Key Predictors**: Payment history (`PAY_0`) and bill amounts are critical for assessing default risk.
- **Feature Correlations**: Strong relationships between payment history, credit limit, and default likelihood.
- **Model Performance**: XGBoost achieves superior classification accuracy with explainable predictions.

## How to Run
1. Clone the notebook to Databricks.
2. Upload the dataset (`credit_card.csv`) to the Databricks file store.
3. Run the notebook sequentially to preprocess data, train models, and interpret results.

