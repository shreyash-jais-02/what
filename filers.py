import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Assuming df is your DataFrame and 'Target' is your target column
X = df.drop(columns='Target')
y = df['Target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
]

# Stacking classifier
clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# Train the stacking classifier
clf.fit(X_train, y_train)

# Function to extract and print feature importances
def print_feature_importances(clf, feature_names):
    # Collect feature importances from base models
    importances = np.zeros(len(feature_names))
    
    for name, model in clf.named_estimators_.items():
        if hasattr(model, 'feature_importances_'):
            importances += model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances += np.abs(model.coef_[0])  # For linear models, take absolute value of coefficients

    # Normalize importances to sum to 1
    importances /= len(clf.named_estimators_)

    # Add importances from final estimator if available
    final_model = clf.final_estimator_
    if hasattr(final_model, 'feature_importances_'):
        final_importances = final_model.feature_importances_
        importances += final_importances
    elif hasattr(final_model, 'coef_'):
        final_importances = np.abs(final_model.coef_[0])
        importances += final_importances

    # Normalize again after adding final estimator importances
    importances /= (len(clf.named_estimators_) + 1)
    
    # Print the feature importances
    for name, importance in zip(feature_names, importances):
        print(f"{name} {importance:.4f}")

# Feature names
feature_names = X.columns.tolist()

# Print the feature importances
print_feature_importances(clf, feature_names)
