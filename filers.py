import pickle
import shap
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load the pickled model
with open('path_to_your_pickled_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your dataset (make sure the features match the model's expected input)
X = pd.read_csv('path_to_your_dataset.csv')  # Adjust this to your data format
y = X['target']  # Adjust to your target column

# Drop the target column to keep only feature columns
X = X.drop(columns=['target'])

# Predict using the model
y_pred = model.predict(X)

# Create the SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Identify true positives and true negatives
cm = confusion_matrix(y, y_pred)
true_positives = (y == 1) & (y_pred == 1)
true_negatives = (y == 0) & (y_pred == 0)

# Extract SHAP values for true positives and true negatives
shap_values_true_positives = shap_values[1][true_positives]
shap_values_true_negatives = shap_values[0][true_negatives]

# Visualization
# Summary plot for true positives
shap.summary_plot(shap_values_true_positives, X[true_positives], plot_type="bar", title="Feature Importance for True Positives")

# Summary plot for true negatives
shap.summary_plot(shap_values_true_negatives, X[true_negatives], plot_type="bar", title="Feature Importance for True Negatives")

# If you want to visualize a single prediction explanation
# For example, the first true positive prediction
shap.force_plot(explainer.expected_value[1], shap_values_true_positives[0], X[true_positives].iloc[0])