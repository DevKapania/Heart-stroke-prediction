"""
Heart Stroke Risk Prediction - Training Script
Author: Dev Kapania | IIT Roorkee Research Intern
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest':       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'xgboost':             XGBClassifier(
                                   n_estimators=200, max_depth=5, learning_rate=0.1,
                                   subsample=0.8, use_label_encoder=False,
                                   eval_metric='logloss', random_state=42
                               )
    }

    trained = {}
    for name, model in models.items():
        print(f'\nTraining {name}...')
        model.fit(X_train, y_train)
        cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
        print(f'  Cross-val F1 (5-fold): {cv_f1:.4f}')
        trained[name] = model

    return trained

def evaluate(models, X_test, y_test):
    print('\n' + '='*60)
    print('MODEL EVALUATION ON TEST SET')
    print('='*60)

    best_name, best_f1 = None, 0
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        print(f'\n{name.upper()}')
        print(f'  F1 Score : {f1:.4f}')
        print(f'  ROC-AUC  : {auc:.4f}')
        print(classification_report(y_test, y_pred,
              target_names=['No Disease', 'Heart Disease']))
        if f1 > best_f1:
            best_f1, best_name = f1, name

    print(f'\n✅ Best Model: {best_name} (F1={best_f1:.4f})')
    return best_name

def save_models(models):
    for name, model in models.items():
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        joblib.dump(model, path)
        print(f'Saved: {path}')

if __name__ == '__main__':
    print('Loading data...')
    X_train, X_test, y_train, y_test = load_data()
    print(f'Train: {X_train.shape} | Test: {X_test.shape}')

    trained_models = train_models(X_train, y_train)
    best = evaluate(trained_models, X_test, y_test)
    save_models(trained_models)

    print('\nTraining complete!')
