#!/usr/bin/env python
"""
Quick start script for ML Hyperparameter Visualizer
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os

def main():
    print("\n" + "="*50)
    print("   ML Hyperparameter Visualizer")
    print("   Streamlit Web Application")
    print("="*50 + "\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Try to import required packages
    try:
        import streamlit
        import sklearn
        import plotly
        print("✓ All required packages are installed")
    except ImportError as e:
        print(f"⚠ Missing package: {e}")
        print("\nInstalling required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("\nStarting ML Hyperparameter Visualizer...")
    print("The application will open in your browser at: http://localhost:8501\n")
    print("Press Ctrl+C to stop the server.\n")
    
    # Start Streamlit app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--logger.level=info"])
    except KeyboardInterrupt:
        print("\n\nApplication stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
