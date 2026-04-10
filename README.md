# ML Hyperparameter Visualizer & Optimizer

A comprehensive web application for visualizing and experimenting with machine learning hyperparameters using scikit-learn algorithms and famous datasets.

## 🎯 Features

- **Multiple Algorithms**: Random Forest, Logistic Regression, Gradient Boosting, SGD, AdaBoost, Linear Regression
- **Real-time Hyperparameter Tuning**: Adjust parameters and instantly see the impact
- **Upload Custom Data**: Use your own CSV files for training
- **Real-World Datasets**: Iris, Wine, Breast Cancer, Digits, Adult Income, California Housing, and synthetic datasets
- **Comprehensive Hyperparameters**: Every algorithm has full control over its hyperparameters
- **Interactive Visualizations**: Decision boundaries, confusion matrices, feature importance plots
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC for classification; MSE, RMSE, MAE, R² for regression
- **Data Distribution Plots**: 2D PCA projections of datasets
- **Model Information**: Complete configuration and hyperparameter details

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🚀 Installation & Setup

### 1. Clone/Navigate to the Project Directory
```bash
cd c:\coding\sklearn_hp
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit scikit-learn pandas numpy matplotlib plotly seaborn
```

### 3. Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 How to Use

1. **Select Task Type**: Choose between Classification or Regression from the sidebar
2. **Choose Data Source**: Either select a built-in dataset or upload your own CSV file
3. **Choose an Algorithm**: Select your desired machine learning algorithm
4. **Adjust Hyperparameters**: Use the sliders and dropdowns to modify algorithm parameters
5. **Train Model**: Click the "Train Model" button to train and visualize results

### Uploading Custom CSV Data

To use your own dataset:
1. Prepare a CSV file with features and a target column
2. The last column should be the target variable
3. Click "Upload CSV File" in the sidebar and select your file
4. The app will automatically process and use your data

Example CSV format:
```
Feature1,Feature2,Feature3,Feature4,Target
1.2,2.5,3.1,4.2,0
2.3,3.1,4.5,5.2,0
3.4,4.2,5.6,6.3,1
...
```

### Hyperparameter Ranges

**Random Forest** (Classification & Regression):
- Number of Trees: 1-500
- Max Depth: 1-50
- Min Samples Split: 2-20
- Min Samples Leaf: 1-10
- Max Features: sqrt, log2, None

**Logistic Regression**:
- Regularization Strength (C): 0.001-100
- Max Iterations: 100-2000
- Solver: lbfgs, liblinear, saga, newton-cg
- Penalty Type: l2, l1, elasticnet

**Linear Regression**:
- Fit Intercept: True/False
- Copy X: True/False
- Force Positive Coefficients: True/False

**Gradient Boosting** (Classification & Regression):
- Number of Estimators: 10-500
- Learning Rate: 0.001-1.0
- Max Depth: 1-15
- Min Samples Split: 2-20
- Min Samples Leaf: 1-10
- Subsample Ratio: 0.1-1.0

**SGD Classifier/Regressor**:
- Loss Function: Multiple options
- Penalty Type: l2, l1, elasticnet, None
- Alpha (Learning Rate): 0.00001-0.1
- Max Iterations: 100-2000

**AdaBoost** (Classification & Regression):
- Number of Estimators: 10-200
- Learning Rate: 0.1-2.0
- Algorithm: SAMME, SAMME.R (Classification only)

## 📊 Visualization Tabs

### 1. Performance Metrics
- Displays key metrics (Accuracy, Precision, Recall, F1, etc.)
- Detailed metrics table

### 2. Decision Boundaries (Classification)
- 2D visualization of decision boundaries
- PCA projection of high-dimensional data
- Predicted vs Actual (Regression)

### 3. Confusion Matrix (Classification)
- Heatmap of prediction results
- Residual plots (Regression)

### 4. Feature Importance
- Top 20 most important features
- Visualization of feature contributions

### 5. Data Distribution
- 2D PCA projection of dataset
- Class distribution visualization

### 6. Model Info
- Algorithm and dataset information
- Hyperparameters used
- Complete model configuration

## 🛠️ Project Structure

```
sklearn_hp/
├── app.py              # Main Streamlit application
├── datasets.py         # Dataset loading utilities
├── models.py           # Model training and evaluation
├── visualizations.py   # Plotting and visualization functions
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 📚 Datasets Available

### Classification:
1. **Iris** - 150 samples, 4 features, 3 classes
2. **Wine** - 178 samples, 13 features, 3 classes
3. **Breast Cancer** - 569 samples, 30 features, 2 classes
4. **Digits** - 1797 samples, 64 features, 10 classes
5. **Make Classification** - 1000 samples, 20 features, 2 classes (synthetic)
6. **Adult (Income)** - Real-world income prediction dataset

### Regression:
1. **Make Regression** - 1000 samples, 20 features (synthetic)
2. **California Housing** - 20640 samples, 8 features, real housing data

### Custom Data:
- Upload any CSV file with your own data

## 🧠 Algorithms Explained

### Classification:
- **Random Forest**: Ensemble of decision trees
- **Logistic Regression**: Linear classification model
- **Gradient Boosting**: Sequentially boosted tree ensemble
- **SGD Classifier**: Stochastic gradient descent classifier
- **AdaBoost**: Adaptive boosting ensemble

### Regression:
- **Linear Regression**: Basic linear model
- **Random Forest Regressor**: Random forest for continuous outputs
- **Gradient Boosting Regressor**: GB trees for regression
- **SGD Regressor**: SGD for regression tasks
- **AdaBoost Regressor**: AdaBoost for regression

## 🎓 Learning Resources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Machine Learning Basics](https://developers.google.com/machine-learning)

## ⚙️ Troubleshooting

**Issue**: "Module not found" error
- **Solution**: Run `pip install -r requirements.txt` again

**Issue**: Application won't start
- **Solution**: Ensure Python 3.8+ is installed, and all dependencies are properly installed

**Issue**: Visualizations not showing
- **Solution**: Check your internet connection (Plotly requires it for some features)

## 🔧 Customization

You can easily extend this application by:
1. Adding new algorithms in `models.py`
2. Adding new datasets in `datasets.py`
3. Creating new visualization types in `visualizations.py`
4. Adding more hyperparameter ranges in `app.py`

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to fork, modify, and improve this project!

## 📧 Support

For issues or questions, please refer to the scikit-learn and Streamlit documentation.

---

**Happy Learning & Experimentation!** 🚀🤖
