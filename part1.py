y_prob_best_thresh = rf_best_thresh.predict_proba(X_test)[:, 1]
y_pred_best_thresh_2 = (y_prob_best_thresh >= best_threshold).astype(int)

true_positives_idx = np.where((y_test == 1) & (y_pred_best_thresh_2 == 1))[0][:5]
true_negatives_idx = np.where((y_test == 0) & (y_pred_best_thresh_2 == 0))[0][:5]

indices = np.concatenate([true_positives_idx, true_negatives_idx])

X_selected = X_test.iloc[indices]

explainer_shap = shap.TreeExplainer(rf_best_thresh)
shap_values = explainer_shap.shap_values(X_selected)

for i, idx in enumerate(indices):
    if i < 5:
        print(f"\nTrue Positive {i+1}")
    else:
        print(f"\nTrue Negative {i-4}")
    
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': shap_values[1][idx]
    }).sort_values(by='importance', ascending=False)
    
    print("Important features:")
    print(feature_importance['feature'].tolist())

explainer_lime = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=[0, 1], discretize_continuous=True)

for i, idx in enumerate(indices):
    explanation = explainer_lime.explain_instance(X_test.iloc[idx].values, rf_best_thresh.predict_proba, num_features=10)
    if i < 5:
        print(f"\nTrue Positive {i+1} LIME Explanation")
    else:
        print(f"\nTrue Negative {i-4} LIME Explanation")
    
    print(explanation.as_list())
