"""Test script for new Kaggle datasets"""
import sys
from datasets import load_dataset, get_available_datasets

def test_new_datasets():
    """Test loading of new Kaggle datasets"""
    
    print("=" * 70)
    print("🌍 Testing New Kaggle Datasets")
    print("=" * 70)
    
    # Test Classification Datasets
    print("\n📊 Classification Datasets (Kaggle):")
    print("-" * 70)
    
    classification_datasets = get_available_datasets("classification")
    new_classification = ["Titanic", "Pima Indians Diabetes"]
    
    for dataset_name in new_classification:
        if dataset_name in classification_datasets:
            print(f"\n✅ Testing: {dataset_name}")
            try:
                X, y, feature_names, target_names = load_dataset(dataset_name)
                print(f"   Shape: {X.shape}")
                print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
                print(f"   Classes: {len(np.unique(y))}")
                print(f"   Feature names: {feature_names[:3]}...")
                print(f"   Target names: {target_names}")
                print(f"   ✅ SUCCESS - Dataset loaded!")
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                return False
        else:
            print(f"❌ {dataset_name} not in classification datasets!")
            return False
    
    # Test Regression Datasets
    print("\n" + "=" * 70)
    print("📈 Regression Datasets (Kaggle):")
    print("-" * 70)
    
    regression_datasets = get_available_datasets("regression")
    new_regression = ["House Prices"]
    
    for dataset_name in new_regression:
        if dataset_name in regression_datasets:
            print(f"\n✅ Testing: {dataset_name}")
            try:
                X, y, feature_names, target_names = load_dataset(dataset_name)
                print(f"   Shape: {X.shape}")
                print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
                print(f"   Target range: {y.min():.0f} - {y.max():.0f}")
                print(f"   Feature names: {feature_names[:3]}...")
                print(f"   ✅ SUCCESS - Dataset loaded!")
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                return False
        else:
            print(f"❌ {dataset_name} not in regression datasets!")
            return False
    
    # Print all available datasets
    print("\n" + "=" * 70)
    print("📋 Complete Dataset List:")
    print("=" * 70)
    print("\n🏷️  Classification Datasets:")
    for i, ds in enumerate(classification_datasets, 1):
        print(f"   {i}. {ds}")
    
    print("\n📊 Regression Datasets:")
    for i, ds in enumerate(regression_datasets, 1):
        print(f"   {i}. {ds}")
    
    print("\n" + "=" * 70)
    print("✨ All new Kaggle datasets tested successfully!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    import numpy as np
    success = test_new_datasets()
    sys.exit(0 if success else 1)
