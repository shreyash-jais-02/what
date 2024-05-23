import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix
import matplotlib.pyplot as plt

# Load training data
df = pd.read_csv('data.csv')

# Replace NaNs with the mean of the respective columns
for col in ['L3M_DR_AMT_MEAN', 'L6M_DR_AMT_MEAN', 'L12M_DR_AMT_MEAN']:
    df[col] = df[col].replace(np.nan, df[col].mean())

# Perform PCA
DR_AMT_MEAN = ['L3M_DR_AMT_MEAN', 'L6M_DR_AMT_MEAN', 'L12M_DR_AMT_MEAN']
pca = PCA(n_components=1)
df['DR_AMT_MEAN_PCA'] = pca.fit_transform(df[DR_AMT_MEAN])

# Assuming similar steps for other PCA components...

# Splitting the data into training and test sets
X = df.drop(columns=['TARGET'])
y = df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# List of columns to use in the model (includes PCA columns)
cols = [...]  # Add the actual column names here

df = df[cols].copy()

# Handling imbalanced dataset
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.9)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X_train1, y_train1 = pipeline.fit_resample(X_train, y_train)

# Random Forest Classifier
RF = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)
RF.fit(X_train1, y_train1)
rf_y_pred_prob = RF.predict_proba(X_test)

# Find the best threshold using the training data
fpr, tpr, thresh = roc_curve(y_test, rf_y_pred_prob[:, 1], pos_label=1)
best_thresh_idx = np.argmax(tpr - fpr)
best_thresh = thresh[best_thresh_idx]

# Load validation data
val_data = pd.read_csv('data2.csv')

# Apply the same preprocessing to the validation data
for col in ['L3M_DR_AMT_MEAN', 'L6M_DR_AMT_MEAN', 'L12M_DR_AMT_MEAN']:
    val_data[col] = val_data[col].replace(np.nan, df[col].mean())  # Use mean from training data

val_data['DR_AMT_MEAN_PCA'] = pca.transform(val_data[DR_AMT_MEAN])

# Assuming similar steps for other PCA components...

X_val = val_data[cols].copy()
y_val = val_data['TARGET']

# Predict on the validation data
rf_y_val_prob = RF.predict_proba(X_val)
rf_y_val_pred = (rf_y_val_prob[:, 1] >= best_thresh).astype(int)

# Calculate recall on the validation data
val_recall = recall_score(y_val, rf_y_val_pred)
val_matrix = confusion_matrix(y_val, rf_y_val_pred)

# Print results
print(f'Validation Recall: {val_recall}')
print(f'Validation Confusion Matrix:\n{val_matrix}')
