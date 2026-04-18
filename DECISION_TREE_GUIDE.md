# 🌳 Decision Tree - Complete Guide

## Overview

Decision Trees are now fully integrated into the ML Hyperparameter Visualizer with comprehensive hyperparameter control to improve accuracy and prevent overfitting. This guide explains how to use Decision Trees effectively.

## Why Decision Trees?

✅ **Interpretable**: Easy to understand how predictions are made
✅ **Visualizable**: Can display the tree structure
✅ **No Scaling Required**: Works with unscaled data
✅ **Mixed Data Types**: Handles both numeric and categorical features
✅ **Non-parametric**: No assumptions about data distribution

## Available Decision Tree Hyperparameters

### 1. **Core Parameters** (Most Important)

#### Max Depth
- **What it does**: Controls tree height
- **Range**: 0-50 (0 = no limit)
- **Effect on Accuracy**:
  - ❌ Too low (1-3): Tree too shallow, underfitting
  - ✅ Optimal (7-15): Balanced fit
  - ❌ Too high/None: Tree too deep, overfitting
- **Recommendation**: Start with 10-15 for most datasets

#### Min Samples Split
- **What it does**: Minimum samples needed to split internal nodes
- **Range**: 2-50
- **Effect on Accuracy**:
  - Lower values → More splits → Higher complexity (overfitting risk)
  - Higher values → Fewer splits → More generalization
- **Recommendation**: Start with 2-10, increase if overfitting

#### Min Samples Leaf
- **What it does**: Minimum samples in each leaf node
- **Range**: 1-20
- **Effect on Accuracy**:
  - Lower values → Smaller leaves → More specific predictions
  - Higher values → Larger leaves → More robust predictions
- **Recommendation**: Start with 1, increase to 5-10 if overfitting

### 2. **Feature Selection**

#### Max Features
- **Options**: 
  - `None`: Use all features
  - `sqrt`: Use sqrt(n_features)
  - `log2`: Use log2(n_features)
  - `auto`: Same as sqrt (sklearn compatibility)
- **Effect**: 
  - Larger → More features considered, higher variance
  - Smaller → Fewer features, higher bias
- **Recommendation**: Start with `None`, try `sqrt` if overfitting

### 3. **Split Criteria** (Classification vs Regression)

#### Classification Criterion
- **gini** (default): Uses Gini impurity, faster
- **entropy**: Information gain-based, slower but sometimes better
- **log_loss**: Probabilistic criterion, good for imbalanced data
- **Recommendation**: Try `gini` first, switch to `entropy` if needed

#### Regression Criterion
- **squared_error** (default): Least squares variation
- **friedman_mse**: Similar to squared error, faster precomputation
- **absolute_error**: Mean absolute error, robust to outliers
- **poisson**: For count data
- **Recommendation**: Use `squared_error` for most tasks

#### Splitter
- **best** (default): Choose best split (thorough)
- **random**: Random split selection (faster, noisier)
- **Recommendation**: Use `best` for better accuracy

### 4. **Regularization Parameters** (Prevent Overfitting)

#### Min Impurity Decrease
- **Range**: 0.0-1.0
- **What it does**: Only split if impurity decrease exceeds this value
- **Higher values** → Fewer splits → Less complex trees
- **Recommendation**: Start with 0.0, increase to 0.01-0.05 if overfitting

#### Cost Complexity Pruning (ccp_alpha)
- **Range**: 0.0-0.1
- **What it does**: Post-pruning parameter - removes less important branches
- **Higher values** → More aggressive pruning
- **Recommendation**: Start with 0.0, try 0.001-0.01 if overfitting

#### Class Weight (Classification Only)
- **Options**:
  - `None`: Equal weight for all classes
  - `balanced`: Auto-adjust weights inversely to class frequency
- **Use Case**: Imbalanced datasets
- **Recommendation**: Use `balanced` for imbalanced data

## Accuracy Best Practices

### For Better Classification Accuracy

1. **Prevent Overfitting**:
   ```
   max_depth = 10-15
   min_samples_split = 5-10
   min_samples_leaf = 2-5
   ccp_alpha = 0.001-0.01
   ```

2. **For Imbalanced Data**:
   ```
   class_weight = balanced
   min_samples_leaf = 10-20
   ```

3. **For Small Datasets** (< 500 samples):
   ```
   max_depth = 5-8
   min_samples_split = 10-20
   min_samples_leaf = 5-10
   ```

4. **For Large Datasets** (> 10000 samples):
   ```
   max_depth = 15-30
   min_samples_split = 2-5
   min_samples_leaf = 1-2
   ```

### For Better Regression Accuracy

1. **Prevent Overfitting**:
   ```
   max_depth = 10-15
   min_samples_split = 5-10
   min_samples_leaf = 2-5
   ```

2. **For Noisy Data**:
   ```
   min_samples_leaf = 10-20
   criterion = absolute_error
   ccp_alpha = 0.001-0.01
   ```

3. **For Count Data (Poisson)**:
   ```
   criterion = poisson
   max_depth = 8-12
   ```

## Common Issues & Solutions

### Issue: 100% Accuracy (Likely Overfitting)
**Solution**:
- Lower `max_depth` from 15 to 5-10
- Increase `min_samples_leaf` from 1 to 5-10
- Increase `ccp_alpha` from 0.0 to 0.001-0.01

### Issue: Very Low Accuracy (Underfitting)
**Solution**:
- Increase `max_depth` from current value
- Decrease `min_samples_split` from 10 to 2-5
- Decrease `min_samples_leaf` from current value
- Decrease `ccp_alpha` if too high

### Issue: Slow Training
**Solution**:
- Increase `min_samples_split`
- Decrease `max_depth`
- Try `min_impurity_decrease = 0.01`

## Example Configurations

### Configuration 1: Iris Dataset (Simple)
```
Max Depth: 5
Min Samples Split: 2
Min Samples Leaf: 1
Max Features: None
Criterion: gini
Class Weight: None
Result: ~95% Accuracy
```

### Configuration 2: Breast Cancer (Medical Classification)
```
Max Depth: 10
Min Samples Split: 5
Min Samples Leaf: 2
Max Features: sqrt
Criterion: entropy
Class Weight: balanced
Result: ~92% Accuracy
```

### Configuration 3: Regression on Synthetic Data
```
Max Depth: 8
Min Samples Split: 5
Min Samples Leaf: 2
Max Features: sqrt
Criterion: squared_error
ccp_alpha: 0.001
Result: R² ~ 0.85
```

## Comparing Decision Tree with Other Algorithms

| Aspect | Decision Tree | Random Forest | Gradient Boosting |
|--------|---------------|---------------|-------------------|
| Speed | ⚡ Very Fast | ⚡⚡ Fast | ⚡⚡⚡ Slow |
| Accuracy (High) | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Interpretability | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| Overfitting Risk | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| Hyperparameters | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**When to use Decision Tree**:
- ✅ Need interpretability
- ✅ Small to medium datasets
- ✅ Quick baseline model
- ✅ Feature importance analysis

## Experiment Workflow

1. **Start Simple**: Use default parameters and check baseline accuracy
   ```
   max_depth = 10
   min_samples_split = 2
   min_samples_leaf = 1
   ```

2. **Adjust for Overfitting**: If accuracy too high, add regularization
   ```
   Increase max_depth → Decrease max_depth
   Increase min_samples_split → Add ccp_alpha
   ```

3. **Swap Criteria**: Try different split criteria
   ```
   Try entropy if gini gives unstable results
   Try absolute_error if regression has outliers
   ```

4. **Fine-tune Features**: Adjust feature selection
   ```
   Try max_features = 'sqrt' or 'log2'
   ```

5. **Polish**: Apply final touches with regularization

## Testing Decision Tree

Run the included test script:
```bash
python test_decision_tree.py
```

This will verify:
- ✅ Decision Tree Classifier with Gini
- ✅ Decision Tree Classifier with Entropy + Regularization
- ✅ Decision Tree Regressor with Squared Error

## Feature Importance Analysis

Decision Trees show which features are most important for splitting. Look for:
- 📊 **Feature Importance** tab in the visualization
- 🔍 See top features driving predictions
- 💡 Use for feature selection in other models

## References

- [scikit-learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Decision Tree Parameters Guide](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Cost Complexity Pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)

---

**Happy Tree Building! 🌳📊**
