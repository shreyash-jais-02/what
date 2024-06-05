import pandas as pd
import numpy as np
from scipy.stats import skew

# Load your dataset
# df = pd.read_csv('your_dataset.csv')

# Define the acceptable range for skewness
acceptable_skewness_range = (-2, 2)

# Detect highly skewed and acceptable skewed columns
highly_skewed_columns = []
acceptable_skewed_columns = []

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    col_skewness = skew(df[col].dropna())
    if col_skewness < acceptable_skewness_range[0] or col_skewness > acceptable_skewness_range[1]:
        highly_skewed_columns.append(col)
    else:
        acceptable_skewed_columns.append(col)

# Log transform highly skewed columns with positive values
for col in highly_skewed_columns:
    if (df[col] > 0).all():
        df[col] = np.log1p(df[col])

# Recalculate skewness after log transformation
highly_skewed_columns_after_log = []
acceptable_skewed_columns_after_log = []

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    col_skewness = skew(df[col].dropna())
    if col_skewness < acceptable_skewness_range[0] or col_skewness > acceptable_skewness_range[1]:
        highly_skewed_columns_after_log.append(col)
    else:
        acceptable_skewed_columns_after_log.append(col)

# Impute highly skewed columns (even after log transform) with median
for col in highly_skewed_columns_after_log:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)

# Impute acceptable skewed columns with mean
for col in acceptable_skewed_columns_after_log:
    mean_value = df[col].mean()
    df[col].fillna(mean_value, inplace=True)

# Display the result
print("Highly skewed columns after log transform:", highly_skewed_columns_after_log)
print("Acceptable skewed columns after log transform:", acceptable_skewed_columns_after_log)
print("Imputation complete.")
