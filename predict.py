"""
Heart Stroke Risk Prediction - Inference Script
Author: Dev Kapania | IIT Roorkee Research Intern

Usage:
    python src/predict.py --input data/sample_patient.csv
"""

import numpy as np
import pandas as pd
import joblib
import argparse
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_model(model_name='xgboost'):
    model_path  = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}\nRun train.py first.')

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess(df, scaler):
    # Feature engineering (same as training)
    df['age_group'] = pd.cut(df['age'], bins=[0,40,55,70,100], labels=[0,1,2,3]).astype(int)
    df['high_chol'] = (df['chol'] > 200).astype(int)
    df['high_bp']   = (df['trestbps'] > 140).astype(int)
    return scaler.transform(df)

def predict(input_path, model_name='xgboost'):
    model, scaler = load_model(model_name)
    df = pd.read_csv(input_path)
    print(f'Loaded {len(df)} patient(s) from {input_path}')

    X = preprocess(df.copy(), scaler)
    predictions  = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = df.copy()
    results['prediction']   = predictions
    results['risk_label']   = results['prediction'].map({0: 'No Disease', 1: 'Heart Disease'})
    results['risk_score_%'] = (probabilities * 100).round(1)

    print('\n' + '='*50)
    print('PREDICTION RESULTS')
    print('='*50)
    for i, row in results.iterrows():
        print(f'Patient {i+1}: {row["risk_label"]} (Risk Score: {row["risk_score_%"]}%)')

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heart Stroke Risk Prediction')
    parser.add_argument('--input', required=True, help='Path to patient CSV file')
    parser.add_argument('--model', default='xgboost', help='Model to use (default: xgboost)')
    args = parser.parse_args()

    results = predict(args.input, args.model)
    output_path = args.input.replace('.csv', '_predictions.csv')
    results.to_csv(output_path, index=False)
    print(f'\nResults saved to: {output_path}')
