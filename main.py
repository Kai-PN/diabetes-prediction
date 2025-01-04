import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn. ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

def predict_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'AUC': roc_auc_score(y_test, y_proba),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }

def process_file(input_file, kfold, model, selection_method, skip_feature_selection):
    try:
        print(f"Processing {input_file}...")
        data = pd.read_csv(input_file)
        print(f"Heading of dataset:")
        print(data.head())
        print(f"Data shape: {data.shape}")
        print(f"Number of non-diabetes and diabetes: {data['target'].value_counts()}")

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        le = LabelEncoder()
        y = le.fit_transform(y)

        skf = StratifiedKFold(n_splits=kfold, random_state=42, shuffle=True)
        rus = RandomUnderSampler()
        fold_metrics_results = []

        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

            if not skip_feature_selection:

                selector = SelectFromModel(selection_method, prefit=False)
                X_train_ft_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
                X_test_ft_selected = selector.transform(X_test)
            else:
                print("Skipping feature selection...")
                X_train_ft_selected = X_train_resampled
                X_test_ft_selected = X_test

            metrics = predict_metrics(model, X_train_ft_selected, X_test_ft_selected, y_train_resampled, y_test)
            fold_metrics_results.append(metrics)

            print(f"Fold {fold + 1} - AUC: {metrics['AUC']:.4f}") 

        fold_metrics_results = pd.DataFrame(fold_metrics_results)
        avg_metrics = fold_metrics_results.mean()

        print(f"Average AUC: {avg_metrics['AUC']:.4f}")
        return avg_metrics
    
    except Exception as e:
        print(f"Error: {e}")

def map_model(model_name):
    if model_name == 'logreg':
        return LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42)
    elif model_name == 'hisgradboost':
        return HistGradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported model {model_name}")

def map_selection_method(method_name):
    if method_name == 'ada':
        return AdaBoostClassifier(estimator=None, algorithm='SAMME', random_state=42)
    elif method_name == 'xgb':
        return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif method_name == 'rf':
        return RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported selection method {method_name}")

def main(input_file, kfold, model_name, method_name, skip_feature_selection):
    model = map_model(model_name)
    selection_method = map_selection_method(method_name)
    avg_metrics = process_file(input_file, kfold, model, selection_method, skip_feature_selection)
    print('\n')
    print("Prediction results:")
    print(f"{avg_metrics}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction Models for Type 2 Diabetes')
    parser.add_argument('--input_file', type=str, help='Dataset file')
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds, default=5')
    parser.add_argument('--model', type=str, default='logreg', help=' logreg : Logistic Regression (default) or hisgradboost: Hist Gradient Boosting')
    parser.add_argument('--selection_method', type=str, default='ada', help='Feature selection method. ada : AdaBoost (default) or xgb : XGBoost')
    parser.add_argument('--skip_feature_selection', action='store_true', help='Skip feature selection')
    args = parser.parse_args()

    main(args.input_file, args.kfold, args.model, args.selection_method, args.skip_feature_selection)