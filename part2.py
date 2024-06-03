import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

with open('pickled_model.pkl', 'rb') as file:
    model = pickle.load(file)

df = pd.read_csv('file.csv')
X = df.drop(columns=['TARGET'])
y = df['TARGET']

predictions = model.predict(X)

true_positives = np.where((predictions == 1) & (y == 1))[0]
true_negatives = np.where((predictions == 0) & (y == 0))[0]
false_positives = np.where((predictions == 1) & (y == 0))[0]
false_negatives = np.where((predictions == 0) & (y == 1))[0]

tp_samples = true_positives[:2]
tn_samples = true_negatives[:2]
misclassified_samples = np.concatenate((false_positives[:1], false_negatives[:1]))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.initjs()
for i in tp_samples:
    shap.force_plot(explainer.expected_value[1], shap_values[1][i], X.iloc[i], matplotlib=True)

for i in tn_samples:
    shap.force_plot(explainer.expected_value[0], shap_values[0][i], X.iloc[i], matplotlib=True)

for i in misclassified_samples:
    shap.force_plot(explainer.expected_value[predictions[i]], shap_values[predictions[i]][i], X.iloc[i], matplotlib=True)

plt.show()
