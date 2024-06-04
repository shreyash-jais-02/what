import pandas as pd

# Assuming X_test is your test DataFrame and y_test, y_pred are your true and predicted labels respectively.
X_test = pd.DataFrame(...)  # Replace with your actual test data
y_test = pd.Series(...)     # Replace with your actual true labels
y_pred = pd.Series(...)     # Replace with your actual predictions

# Identify the indices of true positives, true negatives, and false positives
true_positives = X_test[(y_test == 1) & (y_pred == 1)].index[:2]  # Get first 2 true positives
true_negatives = X_test[(y_test == 0) & (y_pred == 0)].index[:2]  # Get first 2 true negatives
false_positives = X_test[(y_test == 0) & (y_pred == 1)].index[:1] # Get first false positive

# Combine all indices
indices_to_explain = list(true_positives) + list(true_negatives) + list(false_positives)

# Define your explainer and prediction function
explainer = ...  # Initialize your explainer
predict_f_rf = ...  # Define your prediction function

# Loop through selected instances and explain each one
for idx in indices_to_explain:
    choosen_instance = X_test.iloc[[idx]].values[0]
    exp = explainer.explain_instance(choosen_instance, predict_f_rf, num_features=10)
    exp.show_in_notebook(show_all=False)