import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px

sns.set_style("whitegrid")

def plot_decision_boundaries(model, X, y, feature_names):
    """Plot decision boundaries for 2D classification"""
    
    # Reduce to 2D using PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlBu', edgecolors='k', s=50)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('Decision Boundaries (2D PCA Projection)')
    plt.colorbar(scatter, ax=ax, label='Class')
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance for tree-based models"""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(indices)), importances[indices], align='center')
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_ylabel('Feature Importance')
        ax.set_title('Top 20 Feature Importances')
        plt.tight_layout()
        
        return fig
    else:
        return None

def plot_metrics_comparison(metrics_list, algorithm_names):
    """Plot comparison of different metrics"""
    
    fig = go.Figure()
    
    for i, (metrics, algo_name) in enumerate(zip(metrics_list, algorithm_names)):
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig.add_trace(go.Bar(
            name=algo_name,
            x=metric_names,
            y=metric_values
        ))
    
    fig.update_layout(
        title='Model Performance Metrics Comparison',
        xaxis_title='Metrics',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    return fig

def plot_learning_curve(train_scores, test_scores, param_name, param_values):
    """Plot learning curves showing effect of parameter changes"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(param_values), y=list(train_scores),
        name='Training Score',
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=list(param_values), y=list(test_scores),
        name='Validation Score',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title=f'Model Performance vs {param_name}',
        xaxis_title=param_name,
        yaxis_title='Score',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def plot_predicted_vs_actual(y_true, y_pred):
    """Plot predicted vs actual values for regression"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode='markers',
        marker=dict(size=8, opacity=0.6),
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(
        title='Predicted vs Actual Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=500
    )
    
    return fig

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve for binary classification"""
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    
    return fig

def plot_data_distribution(X, y, feature_names, class_names=None):
    """Plot distribution of features"""
    
    X_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=y,
        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Class'},
        title='Data Distribution (2D PCA Projection)',
        height=600
    )
    
    return fig

def plot_decision_tree(model, feature_names, class_names=None, max_depth=None):
    """Plot decision tree structure"""
    from sklearn.tree import plot_tree
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
        max_depth=max_depth
    )
    
    plt.title('Decision Tree Structure', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def get_tree_text_representation(model, feature_names):
    """Get text representation of decision tree"""
    from sklearn.tree import export_text
    
    tree_rules = export_text(model, feature_names=list(feature_names))
    return tree_rules

def plot_tree_depth_complexity(model):
    """Plot tree depth and node complexity statistics"""
    
    # Get tree statistics
    tree = model.tree_
    depth = np.array([0])
    
    def get_tree_depth(node_id, current_depth):
        if tree.feature[node_id] == -2:  # Leaf node
            return current_depth
        else:
            left_depth = get_tree_depth(tree.children_left[node_id], current_depth + 1)
            right_depth = get_tree_depth(tree.children_right[node_id], current_depth + 1)
            return max(left_depth, right_depth)
    
    actual_depth = get_tree_depth(0, 0)
    num_leaves = np.sum(tree.feature == -2)
    num_nodes = tree.node_count
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Tree statistics bar chart
    stats = ['Depth', 'Leaves', 'Nodes']
    values = [actual_depth, num_leaves, num_nodes]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    ax1.bar(stats, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Tree Complexity Statistics', fontsize=14, fontweight='bold')
    for i, v in enumerate(values):
        ax1.text(i, v + max(values)*0.02, str(v), ha='center', fontweight='bold')
    
    # Feature importance from tree
    feature_importance = model.feature_importances_
    feature_names = np.arange(len(feature_importance))
    
    indices = np.argsort(feature_importance)[::-1][:10]
    
    ax2.barh(range(len(indices)), feature_importance[indices], color='#95E1D3', edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(len(indices)))
    ax2.set_yticklabels([f'Feature {i}' for i in indices])
    ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    return fig

def plot_tree_path_analysis(model, X, feature_names, instance_index=0):
    """Analyze and visualize decision path for a specific instance"""
    from sklearn.tree import _tree
    
    decision_path = model.decision_path(X[[instance_index]]).toarray()[0]
    leaf_id = np.argmax(decision_path)
    
    # Get the decision path
    node_index = leaf_id
    path = []
    feature_path = []
    threshold_path = []
    value_path = []
    
    tree = model.tree_
    
    def get_path(node_id):
        if node_id == 0:
            return [(node_id, "Root", -1, -1, tree.value[node_id])]
        
        parent_id = None
        for i in range(tree.node_count):
            if tree.children_left[i] == node_id or tree.children_right[i] == node_id:
                parent_id = i
                break
        
        if parent_id is not None:
            is_left = tree.children_left[parent_id] == node_id
            feature = tree.feature[parent_id]
            threshold = tree.threshold[parent_id]
            direction = "LEFT" if is_left else "RIGHT"
            
            return get_path(parent_id) + [(node_id, direction, feature, threshold, tree.value[node_id])]
        return [(node_id, "Root", -1, -1, tree.value[node_id])]
    
    path_info = get_path(leaf_id)
    
    # Visualize path
    fig, ax = plt.subplots(figsize=(12, 8))
    
    path_text = "Decision Path for Instance #0:\n" + "="*50 + "\n\n"
    for i, (node_id, direction, feature, threshold, value) in enumerate(path_info):
        indent = "  " * i
        if direction == "Root":
            path_text += f"{indent}[Root Node {node_id}]\n"
            path_text += f"{indent}Samples: {int(tree.n_node_samples[node_id])}\n"
        else:
            feature_name = feature_names[feature] if feature >= 0 else "N/A"
            path_text += f"{indent}→ {direction}: {feature_name} {'<=' if direction == 'LEFT' else '>'} {threshold:.4f}\n"
            path_text += f"{indent}  [Node {node_id}] Samples: {int(tree.n_node_samples[node_id])}\n"
    
    path_text += f"\n{'='*50}\n"
    path_text += f"Predicted Class/Value: {tree.value[leaf_id]}\n"
    
    ax.text(0.05, 0.95, path_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.title('Decision Path Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig
