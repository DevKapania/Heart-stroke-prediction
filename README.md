# ❤️ Heart Stroke Risk Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logo=xgboost)
![Tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=tableau&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas)

> A machine learning pipeline that predicts **heart disease risk** in patients using clinical features — with Tableau dashboards for communicating insights to non-technical stakeholders.

---

## 🫀 Project Overview

Cardiovascular disease is the leading cause of death globally. Early prediction and risk stratification can enable preventive care and significantly reduce mortality. This project builds and benchmarks multiple classification models to predict heart stroke risk based on patient clinical data, with a strong focus on **handling class imbalance**, **feature engineering**, and **explainability**.

---

## ⚙️ Features

- ✅ Comprehensive EDA with visual correlation analysis
- ✅ Feature engineering: encoding, scaling, interaction features
- ✅ Class imbalance handling using **SMOTE** (Synthetic Minority Oversampling)
- ✅ Benchmarked 3 models: Logistic Regression, Random Forest, XGBoost
- ✅ Hyperparameter tuning with GridSearchCV + Cross-Validation
- ✅ Evaluation: Precision, Recall, F1-Score, ROC-AUC (focus on minority class)
- ✅ Tableau dashboard for feature importance and risk score visualization

---

## 🗂️ Project Structure

```
heart-stroke-prediction/
│
├── data/
│   ├── raw/                        # Original dataset
│   └── processed/                  # Cleaned & SMOTE-balanced data
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Feature engineering & SMOTE
│   └── 03_modeling.ipynb           # Model training & comparison
│
├── src/
│   ├── preprocess.py               # Preprocessing pipeline
│   ├── train.py                    # Training all 3 models
│   ├── evaluate.py                 # Evaluation metrics & plots
│   └── predict.py                  # Inference on new patient data
│
├── tableau/
│   └── heart_stroke_dashboard.twbx # Tableau workbook
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost_model.pkl
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models Compared

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 78%      | 74%       | 71%    | 72%      | 0.83    |
| Random Forest       | 84%      | 81%       | 79%    | 80%      | 0.89    |
| **XGBoost**         | **87%**  | **85%**   | **83%**| **84%**  | **0.92**|

> ✅ XGBoost selected as final model based on F1-Score and ROC-AUC performance.

---

## 📊 Key Findings from EDA

- Age, cholesterol, resting blood pressure, and max heart rate are the strongest predictors
- Class imbalance: ~85% negative, ~15% positive — addressed using SMOTE
- Strong correlation between ST depression and heart disease risk
- Males in the dataset had higher incidence rates than females

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Training
```bash
python src/train.py
```

### Predict on New Data
```bash
python src/predict.py --input patient_data.csv
```

---

## 📦 Requirements

```
scikit-learn>=1.0
xgboost
pandas
numpy
matplotlib
seaborn
imbalanced-learn   # for SMOTE
joblib
```

---

## 🔄 Pipeline Overview

```
Raw Clinical Data (age, BP, cholesterol, etc.)
            ↓
EDA → Correlation Analysis → Outlier Detection
            ↓
Feature Engineering + Label Encoding + Scaling
            ↓
SMOTE → Balanced Train/Test Split (80/20)
            ↓
Train: Logistic Regression | Random Forest | XGBoost
            ↓
Cross-Validation (5-Fold) + Hyperparameter Tuning
            ↓
Evaluation → Best Model Selection (XGBoost)
            ↓
Tableau Dashboard → Risk Score Visualization
```

---

## 📉 Handling Class Imbalance

Without addressing class imbalance, models tend to predict the majority class (no disease) almost always — achieving high accuracy but poor recall on the minority (disease) class, which is clinically dangerous.

**Solution: SMOTE** — generates synthetic samples of the minority class in feature space, balancing the training set without losing data.

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
```

---

## 📊 Tableau Dashboard

The Tableau dashboard visualizes:
- Feature importance rankings
- Patient risk score distribution
- Age vs. risk correlation
- Model prediction confidence by demographic group

---

## 👤 Author

**Dev Kapania**  
B.Tech CSE (Big Data) — UPES  
Deep Learning Research Intern @ IIT Roorkee  
📧 devkapania2003@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/dev-kapania)

---

## 📄 License

This project is licensed under the MIT License.
