

import shap
import numpy as np
import pandas as pd

# Assuming you have the following variables already defined:
# model: your trained binary classification model
# X: your features dataframe
# y: your actual target values
# predictions: model predictions for X

# Calculate SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Convert SHAP values to a DataFrame for easier analysis
shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)

# Add prediction and actual values to the SHAP DataFrame
shap_values_df['prediction'] = model.predict(X)
shap_values_df['actual'] = y.values

# Filter the DataFrame where prediction == actual == 1
true_positive_shap = shap_values_df[(shap_values_df['prediction'] == 1) & (shap_values_df['actual'] == 1)]

# Filter the DataFrame where prediction == actual == 0
true_negative_shap = shap_values_df[(shap_values_df['prediction'] == 0) & (shap_values_df['actual'] == 0)]

# To interpret the contributions for true positives
print("True Positives - SHAP values")
print(true_positive_shap.head())

# To interpret the contributions for true negatives
print("True Negatives - SHAP values")
print(true_negative_shap.head())

# Visualize a single prediction (e.g., the first true positive case)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[true_positive_shap.index[0]], X.iloc[true_positive_shap.index[0]])