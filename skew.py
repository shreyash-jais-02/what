highly_skewed_columns = []
acceptable_skewed_columns = []

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    col_skewness = skew(df[col].dropna())
    if col_skewness < acceptable_skewness_range[0] or col_skewness > acceptable_skewness_range[1]:
        highly_skewed_columns.append(col)
    else:
        acceptable_skewed_columns.append(col)

# Impute highly skewed columns with median and acceptable skewed columns with mean
for col in highly_skewed_columns:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)

for col in acceptable_skewed_columns:
    mean_value = df[col].mean()
    df[col].fillna(mean_value, inplace=True)
