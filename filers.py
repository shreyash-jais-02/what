import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load your data
# df = pd.read_csv('your_dataset.csv')

# 1. Outlier Removal
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

columns_to_check = df.columns  # Or specify specific columns
df_cleaned = remove_outliers(df, columns_to_check)

# Split data into features and target
X = df_cleaned.drop('target', axis=1)  # Replace 'target' with the actual target column name
y = df_cleaned['target']

# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a Model and Print Feature Importances
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_names = X_train.columns

feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importances)

# 3. Identify and Print Highly Correlated Features
correlation_matrix = df_cleaned.corr().abs()

high_correlation_pairs = []
threshold = 0.8

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] > threshold:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            correlation_value = correlation_matrix.iloc[i, j]
            high_correlation_pairs.append((colname_i, colname_j, correlation_value))

print("High correlation:")
for col1, col2, corr in high_correlation_pairs:
    print(f"{col1}, {col2} (correlation: {corr})")
