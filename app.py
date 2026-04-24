import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset, get_available_datasets
from models import train_model, get_available_algorithms
from visualizations import (
    plot_decision_boundaries, plot_confusion_matrix, plot_feature_importance,
    plot_metrics_comparison, plot_learning_curve, plot_predicted_vs_actual,
    plot_roc_curve, plot_data_distribution, plot_decision_tree, plot_tree_depth_complexity,
    plot_tree_path_analysis, get_tree_text_representation
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="ML Hyperparameter Visualizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 0rem; }
    .metric-card { 
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🤖 ML Hyperparameter Optimizer & Visualizer")
st.markdown("""
    This interactive web application allows you to:
    - Experiment with different ML algorithms from scikit-learn
    - Adjust hyperparameters in real-time
    - See how parameter changes affect model performance
    - Visualize decision boundaries, metrics, and predictions
    - Train on famous datasets or upload your own data
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Task type selection
    task_type = st.radio("Select Task Type", ["Classification", "Regression"])
    task_type_lower = "classification" if task_type == "Classification" else "regression"
    
    # Data source selection
    st.subheader("📊 Data Source")
    data_source = st.radio("Choose Data Source", ["Built-in Datasets", "Upload CSV File"])
    
    uploaded_file = None
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            st.success("✓ File uploaded successfully!")
        dataset_name = "Custom Upload"
    else:
        # Dataset selection
        available_datasets = get_available_datasets(task_type_lower)
        dataset_name = st.selectbox("Choose Dataset", available_datasets)
    
    # Algorithm selection
    st.subheader("🎯 Algorithm")
    available_algorithms = get_available_algorithms(task_type_lower)
    algorithm = st.selectbox("Choose Algorithm", available_algorithms)
    
    # Hyperparameters
    st.subheader("🔧 Hyperparameters")
    hyperparams = {}
    
    if algorithm == "Random Forest":
        hyperparams['n_estimators'] = st.slider("Number of Trees", 1, 500, 100)
        hyperparams['max_depth'] = st.slider("Max Depth", 1, 50, 10)
        hyperparams['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2)
        hyperparams['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 10, 1)
        hyperparams['max_features'] = st.selectbox("Max Features", ['sqrt', 'log2', None])
        
    elif algorithm == "Logistic Regression":
        hyperparams['C'] = st.slider("Regularization Strength (C)", 0.001, 100.0, 1.0, log=True)
        hyperparams['max_iter'] = st.slider("Max Iterations", 100, 2000, 100)
        hyperparams['solver'] = st.selectbox("Solver", ['lbfgs', 'liblinear', 'saga', 'newton-cg'])
        hyperparams['penalty'] = st.selectbox("Penalty Type", ['l2', 'l1', 'elasticnet'])
        
    elif algorithm == "Linear Regression":
        hyperparams['fit_intercept'] = st.checkbox("Fit Intercept", value=True)
        hyperparams['copy_X'] = st.checkbox("Copy X", value=True)
        hyperparams['positive'] = st.checkbox("Force Positive Coefficients", value=False)
        st.info("Linear Regression typically requires minimal hyperparameter tuning")
        
    elif algorithm == "Gradient Boosting":
        hyperparams['n_estimators'] = st.slider("Number of Estimators", 10, 500, 100)
        hyperparams['learning_rate'] = st.slider("Learning Rate", 0.001, 1.0, 0.1, step=0.01)
        hyperparams['max_depth'] = st.slider("Max Depth", 1, 15, 3)
        hyperparams['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2)
        hyperparams['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 10, 1)
        hyperparams['subsample'] = st.slider("Subsample Ratio", 0.1, 1.0, 1.0, step=0.1)
        
    elif algorithm == "SGD Classifier" and task_type_lower == "classification":
        hyperparams['loss'] = st.selectbox("Loss Function", ['hinge', 'log_loss', 'squared_hinge', 'modified_huber'])
        hyperparams['penalty'] = st.selectbox("Penalty Type", ['l2', 'l1', 'elasticnet', None])
        hyperparams['alpha'] = st.slider("Alpha (Learning Rate)", 0.00001, 0.1, 0.0001, log=True)
        hyperparams['max_iter'] = st.slider("Max Iterations", 100, 2000, 1000)
        
    elif algorithm == "SGD Regressor" and task_type_lower == "regression":
        hyperparams['loss'] = st.selectbox("Loss Function", ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'])
        hyperparams['penalty'] = st.selectbox("Penalty Type", ['l2', 'l1', 'elasticnet', None])
        hyperparams['alpha'] = st.slider("Alpha (Learning Rate)", 0.00001, 0.1, 0.0001, log=True)
        hyperparams['max_iter'] = st.slider("Max Iterations", 100, 2000, 1000)
        
    elif algorithm == "AdaBoost":
        hyperparams['n_estimators'] = st.slider("Number of Estimators", 10, 200, 50)
        hyperparams['learning_rate'] = st.slider("Learning Rate", 0.1, 2.0, 1.0, step=0.1)
        if task_type_lower == "classification":
            hyperparams['algorithm'] = st.selectbox("Algorithm", ['SAMME', 'SAMME.R'])
    
    elif algorithm == "Decision Tree":
        st.write("**Core Parameters:**")
        max_depth_value = st.slider("Max Depth (0=Unlimited)", 0, 50, 15)
        hyperparams['max_depth'] = None if max_depth_value == 0 else max_depth_value
        hyperparams['min_samples_split'] = st.slider("Min Samples Split", 2, 50, 2)
        hyperparams['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 20, 1)
        
        st.write("**Feature Selection:**")
        hyperparams['max_features'] = st.selectbox("Max Features", [None, 'sqrt', 'log2', 'auto'])
        
        st.write("**Split Criteria:**")
        if task_type_lower == "classification":
            hyperparams['criterion'] = st.selectbox("Criterion (Split Quality)", ['gini', 'entropy', 'log_loss'])
        else:
            hyperparams['criterion'] = st.selectbox("Criterion (Split Quality)", ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
        
        hyperparams['splitter'] = st.selectbox("Splitter", ['best', 'random'])
        
        st.write("**Regularization:**")
        hyperparams['min_impurity_decrease'] = st.slider("Min Impurity Decrease", 0.0, 1.0, 0.0, step=0.01)
        hyperparams['ccp_alpha'] = st.slider("Cost Complexity Pruning Alpha (ccp_alpha)", 0.0, 0.1, 0.0, step=0.001)
        
        if task_type_lower == "classification":
            hyperparams['class_weight'] = st.selectbox("Class Weight", [None, 'balanced'])
        
        st.info("💡 **Decision Tree Tips:** Lower max_depth and higher min_samples_* prevent overfitting. Use ccp_alpha for pruning.")
    
    # Training settings
    st.subheader("🚀 Training")
    run_button = st.button("Train Model", key="train_button", use_container_width=True)

# Main content
if run_button:
    if data_source == "Upload CSV File" and uploaded_file is None:
        st.error("❌ Please upload a CSV file first!")
    else:
        with st.spinner("Loading and preprocessing data..."):
            # Load dataset
            X, y, feature_names, target_names = load_dataset(dataset_name, uploaded_file)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            st.success("✅ Data loaded successfully!")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Performance Metrics",
            "🎨 Decision Boundaries",
            "🔍 Confusion Matrix",
            "⭐ Feature Importance",
            "📊 Data Distribution",
            "📋 Model Info",
            "🌳 Tree Structure"
        ])
        
        with st.spinner("Training model..."):
            # Train model
            model, X_train, X_test, y_train, y_test, y_pred, metrics = train_model(
                X_scaled, y, algorithm, hyperparams, task_type_lower
            )
            
            st.success("✅ Model trained successfully!")
        
        # Tab 1: Performance Metrics
        with tab1:
            st.subheader("Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            metric_items = list(metrics.items())
            for idx, (metric_name, metric_value) in enumerate(metric_items):
                if metric_value is not None:
                    col = [col1, col2, col3, col4][idx % 4]
                    with col:
                        st.metric(
                            label=metric_name.replace('_', ' ').title(),
                            value=f"{metric_value:.4f}" if isinstance(metric_value, float) else metric_value
                        )
            
            # Detailed metrics table
            st.subheader("Detailed Metrics")
            metrics_df = pd.DataFrame([metrics])
            st.dataframe(metrics_df, use_container_width=True)
        
        # Tab 2: Decision Boundaries
        if task_type_lower == "classification":
            with tab2:
                st.subheader("Decision Boundaries (2D PCA Projection)")
                try:
                    fig = plot_decision_boundaries(model, X_scaled, y, feature_names)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot decision boundaries: {str(e)}")
        else:
            with tab2:
                st.subheader("Predicted vs Actual Values")
                try:
                    fig = plot_predicted_vs_actual(y_test, y_pred)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not plot predictions: {str(e)}")
        
        # Tab 3: Confusion Matrix (Classification only)
        if task_type_lower == "classification":
            with tab3:
                st.subheader("Classification Confusion Matrix")
                try:
                    fig = plot_confusion_matrix(
                        y_test, y_pred,
                        class_names=target_names if target_names is not None else None
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot confusion matrix: {str(e)}")
        else:
            with tab3:
                st.subheader("Residual Analysis")
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_pred, residuals, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residual Plot')
                st.pyplot(fig)
        
        # Tab 4: Feature Importance
        with tab4:
            st.subheader("Feature Importance")
            try:
                fig = plot_feature_importance(model, feature_names)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("This model type doesn't provide feature importance scores.")
            except Exception as e:
                st.error(f"Could not plot feature importance: {str(e)}")
        
        # Tab 5: Data Distribution
        with tab5:
            st.subheader("Dataset Distribution (2D PCA)")
            try:
                fig = plot_data_distribution(X_scaled, y, feature_names, target_names)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not plot data distribution: {str(e)}")
        
        # Tab 6: Model Information
        with tab6:
            st.subheader("Model Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Algorithm Details:**")
                st.write(f"- Algorithm: {algorithm}")
                st.write(f"- Task Type: {task_type}")
                st.write(f"- Dataset: {dataset_name}")
            
            with col2:
                st.write("**Dataset Information:**")
                st.write(f"- Total Samples: {X.shape[0]}")
                st.write(f"- Features: {X.shape[1]}")
                st.write(f"- Train-Test Split: 80-20")
            
            st.write("**Hyperparameters Used:**")
            params_df = pd.DataFrame(
                [(k, v) for k, v in hyperparams.items()],
                columns=["Parameter", "Value"]
            )
            st.dataframe(params_df, use_container_width=True)
            
            # Model string representation
            st.write("**Model Configuration:**")
            st.code(str(model))
        
        # Tab 7: Tree Structure (Decision Tree only)
        with tab7:
            if algorithm == "Decision Tree":
                st.subheader("🌳 Decision Tree Structure Visualization")
                
                tree_viz_options = st.radio(
                    "Select Visualization Type:",
                    ["Full Tree Structure", "Tree Complexity", "Decision Path"],
                    horizontal=True
                )
                
                if tree_viz_options == "Full Tree Structure":
                    st.write("**Tree Structure Diagram**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info("📌 This shows the complete decision tree structure with feature splits, thresholds, and node samples.")
                    with col2:
                        max_depth_display = st.slider("Display Depth Limit", 1, 10, 3)
                    
                    try:
                        fig = plot_decision_tree(
                            model, 
                            feature_names=feature_names,
                            class_names=target_names,
                            max_depth=max_depth_display
                        )
                        st.pyplot(fig, use_container_width=True)
                        st.success("✅ Tree structure displayed successfully!")
                    except Exception as e:
                        st.error(f"Could not visualize tree: {str(e)}")
                
                elif tree_viz_options == "Tree Complexity":
                    st.write("**Tree Complexity & Feature Importance**")
                    st.info("📌 Shows statistics about tree depth, number of leaves and nodes, plus the most important features used in splits.")
                    try:
                        fig = plot_tree_depth_complexity(model)
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not plot tree complexity: {str(e)}")
                
                elif tree_viz_options == "Decision Path":
                    st.write("**Decision Path for Sample Instance**")
                    st.info("📌 Shows the complete decision path taken by the model for a specific instance from root to leaf.")
                    
                    sample_idx = st.slider("Select Instance Index", 0, len(X_test) - 1, 0)
                    
                    try:
                        fig = plot_tree_path_analysis(model, X_test, feature_names, sample_idx)
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not plot decision path: {str(e)}")
                
                # Show tree rules in text format
                with st.expander("📝 View Tree Rules as Text"):
                    try:
                        tree_rules = get_tree_text_representation(model, feature_names)
                        st.code(tree_rules, language="text")
                    except Exception as e:
                        st.warning(f"Could not generate text rules: {str(e)}")
            
            else:
                st.info(f"🚫 Tree structure visualization is only available for Decision Tree models. You selected: {algorithm}")
                st.write("""
                **Why Decision Trees?**
                - Decision Trees create explicit rules that are easy to interpret
                - Visual tree structure shows exactly how the model makes decisions
                - Each node represents a feature split threshold
                - Leaf nodes show the final predictions
                
                **Switch to Decision Tree** to see detailed tree structure visualizations!
                """)

else:
    # Initial state - Show welcome message and instructions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🚀 Getting Started
        
        1. **Choose a Task Type** - Select between Classification or Regression
        2. **Select a Dataset** - Pick from famous datasets or generated data
        3. **Pick an Algorithm** - Choose from various ML algorithms
        4. **Adjust Hyperparameters** - Fine-tune in the sidebar
        5. **Train & Visualize** - Click "Train Model" to see results
        
        ### Available Algorithms (12 Total):
        
        **Classification (6):**
        - Random Forest
        - Logistic Regression
        - Gradient Boosting Classifier
        - SGD Classifier
        - AdaBoost Classifier
        - **Decision Tree** ⭐ (with tree structure visualization!)
        
        **Regression (6):**
        - Linear Regression
        - Random Forest Regressor
        - Gradient Boosting Regressor
        - SGD Regressor
        - AdaBoost Regressor
        - **Decision Tree** ⭐ (with tree structure visualization!)
        
        ### Available Datasets (11 Total):
        
        **Classic Datasets:**
        - Iris, Wine, Breast Cancer, Digits
        - Make Classification, Make Regression
        
        **Real-World Datasets:**
        - Adult (Income) - Income prediction
        - California Housing - Housing prices
        
        **🌍 New Kaggle Datasets:**
        - **🚢 Titanic** - Survival prediction (12,818+ competition teams)
        - **🏥 Pima Diabetes** - Medical diagnosis (789K+ downloads)
        - **🏠 House Prices** - Price prediction (4,881+ competition teams)
        
        **📤 Custom Data:**
        - Upload your own CSV files!
        """)
    
    with col2:
        st.info("""
        ### 🎯 Key Features
        - 12 ML algorithms (6 classification, 6 regression)
        - 11 famous datasets + CSV upload
        - Real-time hyperparameter tuning
        - 7 interactive visualization tabs
        - 🌳 Decision Tree visualization
        - Multiple performance metrics
        - Decision boundaries & ROC curves
        - Tree structure & complexity analysis
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 📚 About This Tool
    This application leverages scikit-learn to provide an intuitive interface for understanding
    how hyperparameters affect machine learning model performance. Includes popular Kaggle competition
    datasets for real-world experience. Perfect for learning and experimentation!
    
    **Latest Addition:** 3 famous Kaggle datasets (Titanic, Pima Diabetes, House Prices)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔧 Built with Streamlit & scikit-learn | Hyperparameter Visualization Tool</p>
</div>
""", unsafe_allow_html=True)
