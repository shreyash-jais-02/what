lfrom sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Example data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

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
clf.fit(X, y)

# Function to extract and print feature importances
def print_feature_importances(clf, feature_names):
    # Collect feature importances from base models
    importances = np.zeros(X.shape[1])
    
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

# Feature names (example)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]

# Print the feature importances
print_feature_importances(clf, feature_names)