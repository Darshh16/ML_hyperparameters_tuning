import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml, load_iris, load_wine, load_breast_cancer, load_digits
import warnings
warnings.filterwarnings('ignore')

def load_dataset(dataset_name, uploaded_file=None):
    """Load a dataset based on the name provided"""
    
    # Handle custom uploaded file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = list(df.columns[:-1])
        target_names = None
        return X, y, feature_names, target_names
    
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
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=5, n_classes=2, random_state=42
        )
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        target_names = np.array([0, 1])
        
    elif dataset_name == "Make Regression":
        X, y = datasets.make_regression(
            n_samples=1000, n_features=20, n_informative=15,
            random_state=42
        )
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        target_names = None
        
    elif dataset_name == "Adult (Income)":
        # Load Adult dataset - predicting income
        try:
            df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                            header=None, na_values='?')
            # Remove rows with missing values
            df = df.dropna()
            # Simple preprocessing - select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X = df[numeric_cols].values
            y = (df[14] == ' >50K').astype(int).values  # Binary classification
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
            target_names = np.array(['<=50K', '>50K'])
        except:
            # Fallback to synthetic data
            X, y = datasets.make_classification(n_samples=500, n_features=10, random_state=42)
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
            target_names = np.array([0, 1])
            
    elif dataset_name == "California Housing":
        # Regression dataset
        try:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            X = data.data
            y = data.target
            feature_names = data.feature_names
            target_names = None
        except:
            X, y = datasets.make_regression(n_samples=500, n_features=8, random_state=42)
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
            target_names = None
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y, feature_names, target_names

def get_available_datasets(algorithm_type="classification"):
    """Return available datasets based on algorithm type"""
    if algorithm_type == "classification":
        return ["Iris", "Wine", "Breast Cancer", "Digits", "Make Classification", "Adult (Income)"]
    else:
        return ["Make Regression", "California Housing"]
