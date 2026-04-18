#!/usr/bin/env python
"""
Test Decision Tree Implementation
"""
from datasets import load_dataset
from models import train_model, get_available_algorithms
from sklearn.preprocessing import StandardScaler
import sys

print("="*70)
print("🌳 DECISION TREE IMPLEMENTATION TEST")
print("="*70)
print()

# Load dataset
print("Loading Iris dataset...")
X, y, features, targets = load_dataset("Iris")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print()

# Show available algorithms
print("Classification Algorithms Available:")
for algo in get_available_algorithms('classification'):
    print(f"  • {algo}")
print()

print("Regression Algorithms Available:")
for algo in get_available_algorithms('regression'):
    print(f"  • {algo}")
print()

# Test Decision Tree Classifier
print("-" * 70)
print("TEST 1: Decision Tree Classifier with Gini Criterion")
print("-" * 70)
hyperparams_clf_gini = {
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,
    'criterion': 'gini',
    'splitter': 'best',
    'class_weight': None,
    'min_impurity_decrease': 0.0,
    'ccp_alpha': 0.0
}

try:
    model, _, _, _, _, _, metrics = train_model(
        X_scaled, y, 'Decision Tree', hyperparams_clf_gini, 'classification'
    )
    print(f"✓ Model trained successfully!")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print()
print("-" * 70)
print("TEST 2: Decision Tree Classifier with Entropy Criterion")
print("-" * 70)
hyperparams_clf_entropy = {
    'max_depth': 7,
    'min_samples_split': 3,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'criterion': 'entropy',
    'splitter': 'best',
    'class_weight': 'balanced',
    'min_impurity_decrease': 0.01,
    'ccp_alpha': 0.001
}

try:
    model, _, _, _, _, _, metrics = train_model(
        X_scaled, y, 'Decision Tree', hyperparams_clf_entropy, 'classification'
    )
    print(f"✓ Model trained successfully!")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print()
print("-" * 70)
print("TEST 3: Decision Tree Regressor with Squared Error")
print("-" * 70)
hyperparams_reg = {
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,
    'criterion': 'squared_error',
    'splitter': 'best',
    'min_impurity_decrease': 0.0,
    'ccp_alpha': 0.0
}

try:
    model, _, _, _, _, _, metrics = train_model(
        X_scaled, y, 'Decision Tree', hyperparams_reg, 'regression'
    )
    print(f"✓ Model trained successfully!")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print()
print("="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print()
print("Decision Tree is now available with:")
print("  ✓ Classification and Regression support")
print("  ✓ 9 comprehensive hyperparameters for fine-tuning")
print("  ✓ Multiple criterion options (Gini, Entropy, etc.)")
print("  ✓ Regularization controls (depth, samples, pruning)")
print("  ✓ Feature selection options (sqrt, log2, all features)")
print()
print("Ready to use with: streamlit run app.py")
print("="*70)
