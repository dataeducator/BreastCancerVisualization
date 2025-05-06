# train_models.py
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load and prepare data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM with best parameters from notebook
best_svm = SVC(
    C=0.1,               # Regularization parameter from your analysis
    gamma=0.01,          # Kernel coefficient from your analysis
    kernel='rbf',        # Radial Basis Function kernel
    probability=True,    # Enable probability estimates
    random_state=42
)
best_svm.fit(X_train_scaled, y_train)

# Train ANN with best architecture from notebook
best_ann = MLPClassifier(
    hidden_layer_sizes=(22, 22, 22),  # 3 hidden layers with 22 nodes each
    activation='relu',                # Rectified Linear Unit activation
    solver='adam',                    # Optimization algorithm
    max_iter=1000,                    # Number of epochs
    random_state=42,
    early_stopping=True,              # Prevent overfitting
    validation_fraction=0.1           # Use 10% of data for validation
)
best_ann.fit(X_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

print("SVM Performance:")
print(evaluate_model(best_svm, X_test_scaled, y_test))

print("\nANN Performance:")
print(evaluate_model(best_ann, X_test_scaled, y_test))

# Save models and scaler
joblib.dump(best_svm, 'svm_model.pkl')
joblib.dump(best_ann, 'ann_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModels saved successfully!")
