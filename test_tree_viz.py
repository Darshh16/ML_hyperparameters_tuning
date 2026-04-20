"""Test script for tree visualization functions"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import sys

# Import visualization functions
from visualizations import (
    plot_decision_tree, plot_tree_depth_complexity, 
    plot_tree_path_analysis, get_tree_text_representation
)

def test_tree_visualizations():
    """Test all tree visualization functions"""
    
    print("=" * 60)
    print("🌳 Testing Decision Tree Visualization Functions")
    print("=" * 60)
    
    # Load data
    print("\n1️⃣ Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train decision tree
    print("2️⃣ Training Decision Tree...")
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        criterion='gini',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    accuracy = model.score(X_test_scaled, y_test)
    print(f"   ✅ Decision Tree trained with accuracy: {accuracy:.4f}")
    
    # Test 1: Full tree structure
    print("\n3️⃣ Testing plot_decision_tree()...")
    try:
        fig = plot_decision_tree(
            model, 
            feature_names=feature_names,
            class_names=target_names,
            max_depth=3
        )
        print("   ✅ plot_decision_tree() works!")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False
    
    # Test 2: Tree complexity
    print("\n4️⃣ Testing plot_tree_depth_complexity()...")
    try:
        fig = plot_tree_depth_complexity(model)
        print("   ✅ plot_tree_depth_complexity() works!")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False
    
    # Test 3: Tree path analysis
    print("\n5️⃣ Testing plot_tree_path_analysis()...")
    try:
        fig = plot_tree_path_analysis(model, X_test_scaled, feature_names, 0)
        print("   ✅ plot_tree_path_analysis() works!")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False
    
    # Test 4: Tree text representation
    print("\n6️⃣ Testing get_tree_text_representation()...")
    try:
        tree_rules = get_tree_text_representation(model, feature_names)
        num_lines = len(tree_rules.split('\n'))
        print(f"   ✅ get_tree_text_representation() works! ({num_lines} lines)")
        print("\n   Sample output (first 5 lines):")
        for line in tree_rules.split('\n')[:5]:
            print(f"      {line}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False
    
    print("\n" + "=" * 60)
    print("✨ All tree visualization tests PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_tree_visualizations()
    sys.exit(0 if success else 1)
