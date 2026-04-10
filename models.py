from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error, confusion_matrix,
    roc_auc_score, roc_curve
)
import numpy as np

def train_model(X, y, algorithm, hyperparams, task_type="classification"):
    """Train a model based on selected algorithm and hyperparameters"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if task_type == "classification":
        if algorithm == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', None),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                max_features=hyperparams.get('max_features', None),
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == "Logistic Regression":
            model = LogisticRegression(
                C=hyperparams.get('C', 1.0),
                max_iter=hyperparams.get('max_iter', 100),
                solver=hyperparams.get('solver', 'lbfgs'),
                penalty=hyperparams.get('penalty', 'l2'),
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 3),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                subsample=hyperparams.get('subsample', 1.0),
                random_state=42
            )
        elif algorithm == "SGD Classifier":
            model = SGDClassifier(
                loss=hyperparams.get('loss', 'hinge'),
                penalty=hyperparams.get('penalty', 'l2'),
                alpha=hyperparams.get('alpha', 0.0001),
                max_iter=hyperparams.get('max_iter', 1000),
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == "AdaBoost":
            model = AdaBoostClassifier(
                n_estimators=hyperparams.get('n_estimators', 50),
                learning_rate=hyperparams.get('learning_rate', 1.0),
                algorithm=hyperparams.get('algorithm', 'SAMME.R'),
                random_state=42
            )
            
    else:  # regression
        if algorithm == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', None),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                max_features=hyperparams.get('max_features', None),
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == "Linear Regression":
            model = LinearRegression(
                fit_intercept=hyperparams.get('fit_intercept', True),
                copy_X=hyperparams.get('copy_X', True),
                positive=hyperparams.get('positive', False)
            )
            
        elif algorithm == "Gradient Boosting":
            model = GradientBoostingRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 3),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                subsample=hyperparams.get('subsample', 1.0),
                random_state=42
            )
        elif algorithm == "SGD Regressor":
            model = SGDRegressor(
                loss=hyperparams.get('loss', 'squared_error'),
                penalty=hyperparams.get('penalty', 'l2'),
                alpha=hyperparams.get('alpha', 0.0001),
                max_iter=hyperparams.get('max_iter', 1000),
                random_state=42
            )
        elif algorithm == "AdaBoost":
            model = AdaBoostRegressor(
                n_estimators=hyperparams.get('n_estimators', 50),
                learning_rate=hyperparams.get('learning_rate', 1.0),
                random_state=42
            )
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {}
    if task_type == "classification":
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = None
    else:
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
    
    return model, X_train, X_test, y_train, y_test, y_pred, metrics

def get_available_algorithms(task_type="classification"):
    """Return available algorithms based on task type"""
    if task_type == "classification":
        return ["Random Forest", "Logistic Regression", "Gradient Boosting", "SGD Classifier", "AdaBoost"]
    else:
        return ["Linear Regression", "Random Forest", "Gradient Boosting", "SGD Regressor", "AdaBoost"]
