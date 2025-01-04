# Diabetes Prediction

This project uses machine learning models to predict the likelihood of diabetes based on behavior and lifestyle profile. It includes preprocessing steps, model training, feature selection, and performance evaluation.

## Table of Contents
- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
- [Usage](#usage)

---

## About the Project

Diabetes is a chronic disease that poses significant health challenges worldwide. This project implements machine learning techniques to analyze patient data and predict diabetes outcomes effectively.

Disclaimer: This is a mini-project for my class of machine learning. Also my first code in github. Welcome for contributions and construction feedbacks. Thank you. 

### Features:
- Preprocessing (random under-sampling for handling imbalanced dataset and feature selection)
- Model support:
  - Logistic Regression (logreg) 
  - Histogram Gradient Boosting Classifier (hisgradboost) (default)
- Feature selection method:
  - AdaBoost (ada) (default)
  - XGBoost (xgb)
  - Random Forest (rf)
- K-fold cross-validation to ensure robust evaluation
  - type(int)
  - default = 5
- Metrics for evaluation:
  - AUC
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - MCC (Matthews correlation coefficient)

---

## Usage
Running the Script

python main.py --input_file <path_to_csv> --kfold 5 --model logreg --selection_method ada

Parameters:
--input_file: Path to your dataset file (CSV format).
--kfold: Number of cross-validation folds (default: 5).
--model: Model to use (logreg, hisgradboost). Default = hisgradboost.
--selection_method: Feature selection method (ada, xgb, rf). Default = ada.

if do not want to run Feature selection, use:
'--skip_feature_selection' instead of '--selection_method' 
e.g. python main.py --input_file <path_to_csv> --kfold 5 --model logreg --skip_feature_selection

### Prerequisites
- Python 3.11.7
- Required Python libraries (listed in `requirements.txt`):
  ```bash
  pandas
  scikit-learn
  matplotlib
  xgboost
  imbalanced-learn

