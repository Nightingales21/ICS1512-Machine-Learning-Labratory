"""
experiment3_loan_regression.py
================================
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 3: Regression Analysis using Linear and Regularized Models

Uses the reusable module ml_lab_utils.py (from Experiment 1) for:
    - EDA                       -> generate_eda_summary()
    - Regression train/eval     -> train_evaluate_regression()
    - Regression metrics        -> regression_performance_metrics()
    - Global plot style         -> set_plot_style()

Dataset: Loan Amount Prediction (Analytics Vidhya / Kaggle "Predict Loan
Amount Data" family), 614 loan applications, mixed numeric/categorical
features, target = LoanAmount (continuous, in thousands).
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      RandomizedSearchCV, KFold, cross_val_score,
                                      cross_validate)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_lab_utils import (set_plot_style, _bold_axis_labels, _save_eps,
                           generate_eda_summary, train_evaluate_regression,
                           regression_performance_metrics)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

FIG_DIR = "figures"
RES_DIR = "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

set_plot_style()

# --------------------------------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------------------------------
df_raw = pd.read_csv("loan_data.csv")
print("Raw dataset shape:", df_raw.shape)
print(df_raw.isnull().sum())

df = df_raw.drop(columns=["Loan_ID"]).copy()

# Drop rows where the TARGET itself is missing -- cannot train/evaluate on
# an unknown target value.
n_before = len(df)
df = df.dropna(subset=["LoanAmount"]).reset_index(drop=True)
print(f"\nDropped {n_before - len(df)} rows with missing target (LoanAmount).")
print("Working dataset shape:", df.shape)

# --------------------------------------------------------------------------
# 2. HANDLE MISSING VALUES (features only, target already clean)
# --------------------------------------------------------------------------
categorical_cols = ["Gender", "Married", "Dependents", "Education",
                     "Self_Employed", "Property_Area", "Loan_Status"]
numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term",
                 "Credit_History"]

for c in categorical_cols:
    df[c] = df[c].fillna(df[c].mode()[0])
for c in numeric_cols:
    df[c] = df[c].fillna(df[c].median())

print("\nMissing values after imputation:", int(df.isnull().sum().sum()))

# --------------------------------------------------------------------------
# 3. ENCODE CATEGORICAL VARIABLES
# --------------------------------------------------------------------------
df_encoded = df.copy()
# Dependents has a "3+" category -> map to numeric ordinal 0/1/2/3
df_encoded["Dependents"] = df_encoded["Dependents"].replace("3+", "3").astype(int)
df_encoded = pd.get_dummies(
    df_encoded,
    columns=["Gender", "Married", "Education", "Self_Employed",
             "Property_Area", "Loan_Status"],
    drop_first=True
)
# Coerce any bool dummy columns to int for downstream numeric operations
bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

print("\nEncoded dataset shape:", df_encoded.shape)
print("Encoded columns:", df_encoded.columns.tolist())

# --------------------------------------------------------------------------
# 4. EDA (reusable function from Experiment 1) -- run on the pre-encoding,
#    imputed dataframe so categorical distributions remain human-readable.
# --------------------------------------------------------------------------
generate_eda_summary(
    df, target_col="LoanAmount", dataset_name="Loan Amount Prediction",
    save_path=f"{FIG_DIR}/eda_loan.eps"
)
plt.close("all")

# --------------------------------------------------------------------------
# 5. ADDITIONAL REQUIRED VISUALIZATIONS
# --------------------------------------------------------------------------
# 5a. Target variable distribution
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(df["LoanAmount"], kde=True, ax=ax, color="#2980b9")
_bold_axis_labels(ax, "Loan Amount (thousands)", "Frequency", "Target Variable Distribution")
_save_eps(fig, f"{FIG_DIR}/target_distribution.eps")
plt.close(fig)

# 5b. Feature vs target scatter plots (top 2 numeric features)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=df["ApplicantIncome"], y=df["LoanAmount"], ax=axes[0],
                 color="#27ae60", s=20)
_bold_axis_labels(axes[0], "Applicant Income", "Loan Amount",
                   "Applicant Income vs Loan Amount")
sns.scatterplot(x=df["CoapplicantIncome"], y=df["LoanAmount"], ax=axes[1],
                 color="#8e44ad", s=20)
_bold_axis_labels(axes[1], "Coapplicant Income", "Loan Amount",
                   "Coapplicant Income vs Loan Amount")
plt.tight_layout()
_save_eps(fig, f"{FIG_DIR}/feature_vs_target_scatter.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 6. TRAIN / TEST SPLIT + FEATURE SCALING
# --------------------------------------------------------------------------
X = df_encoded.drop(columns=["LoanAmount"]).values
y = df_encoded["LoanAmount"].values
feature_names = df_encoded.drop(columns=["LoanAmount"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------
# 7. BASELINE LINEAR REGRESSION (via reusable train_evaluate_regression)
# --------------------------------------------------------------------------
baseline_models = {"Linear Regression": LinearRegression()}
baseline_results_df, baseline_fitted = train_evaluate_regression(
    baseline_models, X_train_scaled, X_test_scaled, y_train, y_test, scale=False
)
print("\n=== Baseline Linear Regression ===")
print(baseline_results_df)

# --------------------------------------------------------------------------
# 8. HYPERPARAMETER TUNING: RIDGE, LASSO, ELASTIC NET
#    (5-fold CV, both GridSearchCV and RandomizedSearchCV)
# --------------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search_spaces = {
    "Ridge Regression": (Ridge(random_state=RANDOM_STATE),
                          {"alpha": [0.01, 0.1, 1, 10, 100]}),
    "Lasso Regression": (Lasso(random_state=RANDOM_STATE, max_iter=10000),
                          {"alpha": [0.001, 0.01, 0.1, 1, 10]}),
    "Elastic Net Regression": (ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
                                {"alpha": [0.01, 0.1, 1, 10],
                                 "l1_ratio": [0.2, 0.5, 0.8]}),
}

tuning_summary_rows = []
best_models = {"Linear Regression": baseline_fitted["Linear Regression"]}

for name, (estimator, grid) in search_spaces.items():
    # GridSearchCV
    t0 = time.perf_counter()
    gs = GridSearchCV(estimator, grid, cv=kf, scoring="r2", n_jobs=-1)
    gs.fit(X_train_scaled, y_train)
    grid_time = time.perf_counter() - t0

    # RandomizedSearchCV
    n_combinations = np.prod([len(v) for v in grid.values()])
    n_iter = min(10, int(n_combinations))
    t0 = time.perf_counter()
    rs = RandomizedSearchCV(estimator, grid, cv=kf, scoring="r2",
                             n_iter=n_iter, random_state=RANDOM_STATE, n_jobs=-1)
    rs.fit(X_train_scaled, y_train)
    random_time = time.perf_counter() - t0

    # Use GridSearchCV's result as the "official" tuned model (exhaustive,
    # deterministic); RandomizedSearchCV result reported for comparison only.
    tuning_summary_rows.append({
        "Model": name,
        "Search Method": "GridSearchCV",
        "Best Parameters": str(gs.best_params_),
        "Best CV R2": gs.best_score_,
        "Execution Time (s)": grid_time,
    })
    tuning_summary_rows.append({
        "Model": name,
        "Search Method": "RandomizedSearchCV",
        "Best Parameters": str(rs.best_params_),
        "Best CV R2": rs.best_score_,
        "Execution Time (s)": random_time,
    })
    best_models[name] = gs.best_estimator_

tuning_summary_df = pd.DataFrame(tuning_summary_rows).set_index(["Model", "Search Method"])
tuning_summary_df.to_csv(f"{RES_DIR}/hyperparameter_tuning_summary.csv")
print("\n=== Hyperparameter Tuning Summary ===")
print(tuning_summary_df)

# --------------------------------------------------------------------------
# 9. 5-FOLD CROSS-VALIDATION PERFORMANCE (all 4 models, on training data)
# --------------------------------------------------------------------------
scoring = {
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2",
}

cv_rows = []
for name, model in best_models.items():
    cv_res = cross_validate(model, X_train_scaled, y_train, cv=kf, scoring=scoring)
    mae = -cv_res["test_MAE"].mean()
    mse = -cv_res["test_MSE"].mean()
    rmse = np.sqrt(mse)
    r2 = cv_res["test_R2"].mean()
    cv_rows.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})

cv_performance_df = pd.DataFrame(cv_rows).set_index("Model")
cv_performance_df.to_csv(f"{RES_DIR}/cross_validation_performance.csv")
print("\n=== 5-Fold Cross-Validation Performance ===")
print(cv_performance_df)

# --------------------------------------------------------------------------
# 10. TEST SET PERFORMANCE (all 4 models, reusable metrics function)
# --------------------------------------------------------------------------
test_rows = []
test_predictions = {}
train_times = {}
for name, model in best_models.items():
    t0 = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    train_t = time.perf_counter() - t0
    y_pred = model.predict(X_test_scaled)
    test_predictions[name] = y_pred
    train_times[name] = train_t
    metrics = regression_performance_metrics(y_test, y_pred, model_name=name,
                                               verbose=True, return_dict=True)
    metrics["Training Time (s)"] = train_t
    test_rows.append(metrics)

test_performance_df = pd.DataFrame(test_rows).set_index("Model")
test_performance_df.to_csv(f"{RES_DIR}/test_set_performance.csv")
print("\n=== Test Set Performance ===")
print(test_performance_df)

best_model_name = test_performance_df["R2"].idxmax()
print(f"\nBest model on test R2: {best_model_name}")

# --------------------------------------------------------------------------
# 11. VISUALIZATIONS: Predicted vs Actual, Residuals, Coefficients
# --------------------------------------------------------------------------
# 11a. Predicted vs Actual (2x2 grid, all four models)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for ax, (name, y_pred) in zip(axes, test_predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, color="#2980b9")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, linestyle="--", color="#c0392b", linewidth=1.5)
    _bold_axis_labels(ax, "Actual Loan Amount", "Predicted Loan Amount", name)
plt.tight_layout()
_save_eps(fig, f"{FIG_DIR}/predicted_vs_actual.eps")
plt.close(fig)

# 11b. Residual plots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for ax, (name, y_pred) in zip(axes, test_predictions.items()):
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color="#e67e22")
    ax.axhline(0, linestyle="--", color="#c0392b", linewidth=1.5)
    _bold_axis_labels(ax, "Predicted Loan Amount", "Residual", name)
plt.tight_layout()
_save_eps(fig, f"{FIG_DIR}/residual_plots.eps")
plt.close(fig)

# 11c. Training error vs validation error (learning-curve-style, via CV folds)
train_val_rows = []
for name, model in best_models.items():
    cv_res = cross_validate(model, X_train_scaled, y_train, cv=kf,
                             scoring="neg_mean_squared_error",
                             return_train_score=True)
    train_rmse = np.sqrt(-cv_res["train_score"]).mean()
    val_rmse = np.sqrt(-cv_res["test_score"]).mean()
    train_val_rows.append({"Model": name, "Train RMSE": train_rmse, "Validation RMSE": val_rmse})
train_val_df = pd.DataFrame(train_val_rows).set_index("Model")
train_val_df.to_csv(f"{RES_DIR}/train_vs_validation_error.csv")
print("\n=== Training vs Validation Error ===")
print(train_val_df)

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(train_val_df))
width = 0.35
ax.bar(x_pos - width/2, train_val_df["Train RMSE"], width, label="Training RMSE", color="#2980b9")
ax.bar(x_pos + width/2, train_val_df["Validation RMSE"], width, label="Validation RMSE", color="#c0392b")
ax.set_xticks(x_pos)
ax.set_xticklabels(train_val_df.index, rotation=20, ha="right")
ax.legend()
_bold_axis_labels(ax, "Model", "RMSE", "Training Error vs Validation Error")
_save_eps(fig, f"{FIG_DIR}/train_vs_validation_error.eps")
plt.close(fig)

# 11d. Coefficient comparison bar plot
coef_data = {}
for name, model in best_models.items():
    coef_data[name] = model.coef_
coef_df = pd.DataFrame(coef_data, index=feature_names)
coef_df.to_csv(f"{RES_DIR}/coefficient_comparison.csv")
print("\n=== Coefficient Comparison (first 5 features) ===")
print(coef_df.head())

top_features = coef_df.abs().sum(axis=1).sort_values(ascending=False).head(8).index
fig, ax = plt.subplots(figsize=(10, 6))
coef_df.loc[top_features].plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.legend(prop=fm.FontProperties(family="Times New Roman", size=11))
_bold_axis_labels(ax, "Coefficient Value", "Feature", "Coefficient Comparison (Top 8 Features)")
_save_eps(fig, f"{FIG_DIR}/coefficient_comparison.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 12. REGULARIZATION PATH / OVERFITTING-UNDERFITTING ANALYSIS
# --------------------------------------------------------------------------
# Ridge: effect of alpha on train/val RMSE
ridge_alphas = [0.01, 0.1, 1, 10, 100]
ridge_path_rows = []
for a in ridge_alphas:
    model = Ridge(alpha=a, random_state=RANDOM_STATE)
    cv_res = cross_validate(model, X_train_scaled, y_train, cv=kf,
                             scoring="neg_mean_squared_error", return_train_score=True)
    ridge_path_rows.append({
        "alpha": a,
        "Train RMSE": np.sqrt(-cv_res["train_score"]).mean(),
        "Validation RMSE": np.sqrt(-cv_res["test_score"]).mean(),
    })
ridge_path_df = pd.DataFrame(ridge_path_rows).set_index("alpha")
ridge_path_df.to_csv(f"{RES_DIR}/ridge_alpha_path.csv")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ridge_path_df.index, ridge_path_df["Train RMSE"], marker="o",
        label="Training RMSE", color="#2980b9")
ax.plot(ridge_path_df.index, ridge_path_df["Validation RMSE"], marker="s",
        label="Validation RMSE", color="#c0392b")
ax.set_xscale("log")
ax.legend()
_bold_axis_labels(ax, "Alpha (log scale)", "RMSE", "Ridge: Effect of Regularization Strength")
_save_eps(fig, f"{FIG_DIR}/ridge_alpha_path.eps")
plt.close(fig)

# Lasso: number of non-zero coefficients vs alpha (feature sparsity)
lasso_alphas = [0.001, 0.01, 0.1, 1, 10]
lasso_sparsity_rows = []
for a in lasso_alphas:
    model = Lasso(alpha=a, random_state=RANDOM_STATE, max_iter=10000)
    model.fit(X_train_scaled, y_train)
    n_nonzero = int(np.sum(model.coef_ != 0))
    lasso_sparsity_rows.append({"alpha": a, "Non-zero Coefficients": n_nonzero})
lasso_sparsity_df = pd.DataFrame(lasso_sparsity_rows).set_index("alpha")
lasso_sparsity_df.to_csv(f"{RES_DIR}/lasso_sparsity.csv")
print("\n=== Lasso Feature Sparsity vs Alpha ===")
print(lasso_sparsity_df)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(lasso_sparsity_df.index, lasso_sparsity_df["Non-zero Coefficients"],
        marker="o", color="#8e44ad")
ax.set_xscale("log")
_bold_axis_labels(ax, "Alpha (log scale)", "Non-zero Coefficients",
                   "Lasso: Feature Sparsity vs Regularization Strength")
_save_eps(fig, f"{FIG_DIR}/lasso_sparsity.eps")
plt.close(fig)

print("\nAll figures saved under:", os.path.abspath(FIG_DIR))
print("All result tables saved under:", os.path.abspath(RES_DIR))
print("\nDone.")
