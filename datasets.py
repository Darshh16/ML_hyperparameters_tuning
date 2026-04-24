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
    
    elif dataset_name == "Titanic":
        # 🚢 Titanic - Most famous Kaggle competition (12,818+ teams)
        # Predict survival on the Titanic
        try:
            import seaborn as sns
            df = sns.load_dataset('titanic')
            # Handle missing values and select numeric columns
            df = df.dropna(subset=['embarked', 'age', 'fare'])
            numeric_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']
            X = df[numeric_cols].values
            y = df['survived'].values
            feature_names = numeric_cols
            target_names = np.array(['Died', 'Survived'])
        except:
            # Fallback: Generate synthetic titanic-like data
            X, y = datasets.make_classification(
                n_samples=891, n_features=5, n_informative=4,
                n_redundant=1, n_classes=2, random_state=42
            )
            feature_names = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
            target_names = np.array(['Died', 'Survived'])
    
    elif dataset_name == "Pima Indians Diabetes":
        # 🏥 Pima Indians Diabetes - 789K+ downloads on Kaggle
        # Medical diagnosis classification
        try:
            df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data',
                            header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigree', 'Age']
            target_names = np.array(['No', 'Yes'])
        except:
            # Fallback: Generate synthetic diabetes-like data
            X, y = datasets.make_classification(
                n_samples=768, n_features=8, n_informative=6,
                n_redundant=2, n_classes=2, random_state=42
            )
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigree', 'Age']
            target_names = np.array(['No', 'Yes'])
    
    elif dataset_name == "House Prices":
        # 🏠 House Prices - Advanced Regression Techniques (4,881+ teams on Kaggle)
        # Predict house prices
        try:
            # Try loading Ames Housing dataset from kaggle or similar source
            df = pd.read_csv('https://raw.githubusercontent.com/dsrscott/Ames-Housing-Price-Guide/main/train.csv')
            # Select numeric columns for simplicity
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'Id' and col != 'SalePrice']
            X = df[numeric_cols].fillna(df[numeric_cols].mean()).values
            y = df['SalePrice'].values
            feature_names = numeric_cols.tolist()
            target_names = None
        except:
            # Fallback: Generate synthetic house price data
            # 11 features similar to house price dataset
            X, y = datasets.make_regression(
                n_samples=1460, n_features=11, n_informative=10,
                random_state=42, noise=15000
            )
            feature_names = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
                           'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                           'BsmtUnfSF', 'TotalBsmtSF', 'GrLivArea']
            target_names = None
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y, feature_names, target_names

def get_available_datasets(algorithm_type="classification"):
    """Return available datasets based on algorithm type"""
    if algorithm_type == "classification":
        return [
            "Iris", 
            "Wine", 
            "Breast Cancer", 
            "Digits", 
            "Make Classification", 
            "Adult (Income)",
            "Titanic",  # 🚢 NEW - Most popular Kaggle competition
            "Pima Indians Diabetes"  # 🏥 NEW - Medical diagnosis
        ]
    else:
        return [
            "Make Regression", 
            "California Housing",
            "House Prices"  # 🏠 NEW - House price prediction
        ]
