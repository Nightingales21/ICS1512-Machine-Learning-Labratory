"""
demo_experiment1.py
====================
Demonstration for Experiment 1 (ICS1512).
Exercises the reusable ml_lab_utils functions against several standard
sklearn datasets that stand in for the datasets referenced in the manual:

    Iris                -> Iris Dataset (classification)
    Breast Cancer       -> Predicting Diabetes-style binary classification
    Diabetes            -> Loan-Amount-Prediction-style regression
    Digits (8x8)        -> Handwritten Character Recognition / MNIST proxy

Each dataset is run through:
    1. generate_eda_summary()              -> one 12-panel EDA figure (.eps, 600dpi)
    2. train_evaluate_classification() /
       train_evaluate_regression()          -> reusable model training+eval
    3. classification_performance_metrics() /
       regression_performance_metrics()     -> already invoked internally
"""

import sys
sys.path.append(".")
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ml_lab_utils import (generate_eda_summary, train_evaluate_classification,
                           train_evaluate_regression)

FIG_DIR = "../figures"

# ---------------------------------------------------------------------
# 1) IRIS  -> Classification demo
# ---------------------------------------------------------------------
print("=" * 70, "\nIRIS DATASET (Classification)\n", "=" * 70)
iris = load_iris(as_frame=True)
df_iris = iris.frame.copy()
df_iris["target"] = df_iris["target"].map(dict(enumerate(iris.target_names)))
generate_eda_summary(df_iris, target_col="target", dataset_name="Iris",
                      save_path=f"{FIG_DIR}/eda_iris.eps")

X = iris.data.values
y = iris.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM (RBF)": SVC(probability=True, random_state=42),
}
iris_results, _ = train_evaluate_classification(clf_models, X_train, X_test,
                                                  y_train, y_test, scale=True)
print("\nIris classification summary:\n", iris_results, "\n")

# ---------------------------------------------------------------------
# 2) BREAST CANCER -> Binary classification demo (Predicting Diabetes proxy)
# ---------------------------------------------------------------------
print("=" * 70, "\nBREAST CANCER DATASET (Classification)\n", "=" * 70)
bc = load_breast_cancer(as_frame=True)
df_bc = bc.frame.copy()
df_bc["target"] = df_bc["target"].map({0: "malignant", 1: "benign"})
generate_eda_summary(df_bc, target_col="target", dataset_name="Breast Cancer",
                      save_path=f"{FIG_DIR}/eda_breast_cancer.eps")

X = bc.data.values
y = bc.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)
bc_results, _ = train_evaluate_classification(clf_models, X_train, X_test,
                                                y_train, y_test, scale=True,
                                                verbose=False)
print("\nBreast Cancer classification summary:\n", bc_results, "\n")

# ---------------------------------------------------------------------
# 3) DIABETES -> Regression demo (Loan Amount Prediction proxy)
# ---------------------------------------------------------------------
print("=" * 70, "\nDIABETES DATASET (Regression)\n", "=" * 70)
diab = load_diabetes(as_frame=True)
df_diab = diab.frame.copy()
generate_eda_summary(df_diab, target_col="target", dataset_name="Diabetes",
                      save_path=f"{FIG_DIR}/eda_diabetes.eps")

X = diab.data.values
y = diab.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
}
diab_results, _ = train_evaluate_regression(reg_models, X_train, X_test,
                                             y_train, y_test, scale=True)
print("\nDiabetes regression summary:\n", diab_results, "\n")

# ---------------------------------------------------------------------
# 4) DIGITS -> Handwritten character recognition / MNIST proxy
# ---------------------------------------------------------------------
print("=" * 70, "\nDIGITS DATASET (Classification / MNIST proxy)\n", "=" * 70)
digits = load_digits(as_frame=True)
df_digits = digits.frame.copy()
generate_eda_summary(df_digits, target_col="target", dataset_name="Digits (MNIST proxy)",
                      save_path=f"{FIG_DIR}/eda_digits.eps")

X = digits.data.values
y = digits.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)
digits_results, _ = train_evaluate_classification(
    {"Logistic Regression": LogisticRegression(max_iter=2000),
     "Random Forest": RandomForestClassifier(random_state=42),
     "KNN": KNeighborsClassifier()},
    X_train, X_test, y_train, y_test, scale=True, verbose=False)
print("\nDigits classification summary:\n", digits_results, "\n")

print("All EDA figures saved to:", FIG_DIR)
