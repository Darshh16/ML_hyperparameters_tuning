import numpy as np
import pandas as pd
from sklearn import datasets

def load_dataset(dataset_name):
    """Load a dataset based on the name provided"""
    
    if dataset_name == "Iris":
        data = datasets.load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
    elif dataset_name == "Wine":
        data = datasets.load_wine()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = np.array(['Malignant', 'Benign'])
        
    elif dataset_name == "Digits":
        data = datasets.load_digits()
        X = data.data
        y = data.target
        feature_names = [f"Pixel {i}" for i in range(X.shape[1])]
        target_names = data.target_names
        
    elif dataset_name == "Make Classification":
        X, y = datasets.make_classification(
            n_samples=500, n_features=20, n_informative=15,
            n_redundant=5, n_classes=2, random_state=42
        )
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        target_names = np.array([0, 1])
        
    elif dataset_name == "Make Regression":
        X, y = datasets.make_regression(
            n_samples=500, n_features=20, n_informative=15,
            random_state=42
        )
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        target_names = None
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y, feature_names, target_names

def get_available_datasets(algorithm_type="classification"):
    """Return available datasets based on algorithm type"""
    if algorithm_type == "classification":
        return ["Iris", "Wine", "Breast Cancer", "Digits", "Make Classification"]
    else:
        return ["Make Regression"]
