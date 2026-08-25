"""
experiment4_spambase_logreg_svm.py
====================================
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 4: Binary Classification using Linear and Kernel-Based Models
(Logistic Regression and Support Vector Machine)

Uses the reusable module ml_lab_utils.py (from Experiment 1) for:
    - EDA                         -> generate_eda_summary()
    - Classification train/eval   -> train_evaluate_classification()
    - Classification metrics      -> classification_performance_metrics()
    - Global plot style           -> set_plot_style()

Dataset: Spambase (UCI ML Repository / Kaggle mirror), 4601 emails x 57
features + binary target (1 = spam, 0 = ham). Same dataset and column
naming as Experiment 2.
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
                                      RandomizedSearchCV, StratifiedKFold,
                                      cross_validate)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix)

from ml_lab_utils import (set_plot_style, _bold_axis_labels, _save_eps,
                           generate_eda_summary, train_evaluate_classification,
                           classification_performance_metrics)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

FIG_DIR = "figures"
RES_DIR = "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

set_plot_style()

# --------------------------------------------------------------------------
# 1. LOAD DATASET (same column naming as Experiment 2)
# --------------------------------------------------------------------------
COLUMN_NAMES = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_semicolon", "char_freq_paren", "char_freq_bracket",
    "char_freq_bang", "char_freq_dollar", "char_freq_pound",
    "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total", "spam",
]

raw = pd.read_csv("spambase.csv")
raw.columns = COLUMN_NAMES
df = raw.copy()
print("Dataset shape:", df.shape)
print(df["spam"].value_counts())

# --------------------------------------------------------------------------
# 2. HANDLE MISSING VALUES (none expected, verified defensively)
# --------------------------------------------------------------------------
n_missing = int(df.isnull().sum().sum())
print(f"Missing cells: {n_missing}")
if n_missing > 0:
    df = df.fillna(df.median(numeric_only=True))

# --------------------------------------------------------------------------
# 3. EDA (reusable function from Experiment 1)
# --------------------------------------------------------------------------
generate_eda_summary(
    df, target_col="spam", dataset_name="Spambase (Exp. 4)",
    save_path=f"{FIG_DIR}/eda_spambase.eps"
)
plt.close("all")

# --------------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT + FEATURE STANDARDIZATION
# --------------------------------------------------------------------------
X = df.drop(columns=["spam"]).values
y = df["spam"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def time_fit_predict(model, Xtr, Xte, ytr):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = model.predict(Xte)
    pred_t = time.perf_counter() - t0
    return y_pred, train_t, pred_t


# --------------------------------------------------------------------------
# 5. BASELINE LOGISTIC REGRESSION (via reusable train_evaluate_classification)
# --------------------------------------------------------------------------
baseline_models = {"Logistic Regression (Baseline)": LogisticRegression(max_iter=2000)}
baseline_results_df, baseline_fitted = train_evaluate_classification(
    baseline_models, X_train_scaled, X_test_scaled, y_train, y_test, scale=False
)
print("\n=== Baseline Logistic Regression ===")
print(baseline_results_df)

# --------------------------------------------------------------------------
# 6. LOGISTIC REGRESSION HYPERPARAMETER TUNING (GridSearchCV + RandomizedSearchCV)
# --------------------------------------------------------------------------
logreg_param_grid = [
    {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear"]},
    {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100], "solver": ["saga"]},
]

print("[LogReg] Starting GridSearchCV...", flush=True)
t0 = time.perf_counter()
logreg_grid = GridSearchCV(LogisticRegression(max_iter=1000), logreg_param_grid,
                            cv=cv_strategy, scoring="accuracy", n_jobs=1)
logreg_grid.fit(X_train_scaled, y_train)
logreg_grid_time = time.perf_counter() - t0
print(f"[LogReg] GridSearchCV done in {logreg_grid_time:.1f}s", flush=True)

print("[LogReg] Starting RandomizedSearchCV...", flush=True)
t0 = time.perf_counter()
logreg_random = RandomizedSearchCV(LogisticRegression(max_iter=1000), logreg_param_grid,
                                    cv=cv_strategy, scoring="accuracy", n_iter=10,
                                    random_state=RANDOM_STATE, n_jobs=1)
logreg_random.fit(X_train_scaled, y_train)
logreg_random_time = time.perf_counter() - t0
print(f"[LogReg] RandomizedSearchCV done in {logreg_random_time:.1f}s", flush=True)

logreg_tuning_summary = pd.DataFrame({
    "GridSearchCV": {
        "Best Parameters": str(logreg_grid.best_params_),
        "Best CV Accuracy": logreg_grid.best_score_,
        "Execution Time (s)": logreg_grid_time,
    },
    "RandomizedSearchCV": {
        "Best Parameters": str(logreg_random.best_params_),
        "Best CV Accuracy": logreg_random.best_score_,
        "Execution Time (s)": logreg_random_time,
    },
})
logreg_tuning_summary.to_csv(f"{RES_DIR}/logreg_tuning_summary.csv")
print("\n=== Logistic Regression Tuning Summary ===")
print(logreg_tuning_summary)

best_logreg = logreg_grid.best_estimator_
y_pred_lr, lr_train_t, lr_pred_t = time_fit_predict(best_logreg, X_train_scaled,
                                                      X_test_scaled, y_train)
y_proba_lr = best_logreg.predict_proba(X_test_scaled)
logreg_metrics = classification_performance_metrics(
    y_test, y_pred_lr, y_proba_lr, model_name="Logistic Regression (Tuned)",
    return_dict=True, plot=True, save_path=f"{FIG_DIR}/cm_logreg.eps"
)
logreg_metrics["Training Time (s)"] = lr_train_t
print("\n=== Tuned Logistic Regression Performance ===")
print(logreg_metrics)

# --------------------------------------------------------------------------
# 7. SVM: TRAIN WITH DIFFERENT KERNELS (baseline, default hyperparameters)
# --------------------------------------------------------------------------
kernels = ["linear", "poly", "rbf", "sigmoid"]
svm_kernel_rows = []
svm_kernel_models = {}
for kernel in kernels:
    print(f"[SVM baseline] Training kernel={kernel}...", flush=True)
    svm = SVC(kernel=kernel, probability=False, random_state=RANDOM_STATE)
    y_pred, train_t, pred_t = time_fit_predict(svm, X_train_scaled, X_test_scaled, y_train)
    svm_kernel_rows.append({
        "Kernel": kernel,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Training Time (s)": train_t,
    })
    svm_kernel_models[kernel] = svm

svm_kernel_df = pd.DataFrame(svm_kernel_rows).set_index("Kernel")
svm_kernel_df.to_csv(f"{RES_DIR}/svm_kernel_comparison.csv")
print("\n=== SVM Kernel-wise Performance ===")
print(svm_kernel_df)

best_kernel = svm_kernel_df["Accuracy"].idxmax()
print(f"\nBest SVM kernel (baseline hyperparameters): {best_kernel}")

fig, ax = plt.subplots(figsize=(8, 5))
svm_kernel_df["Accuracy"].plot(kind="bar", ax=ax, color="#16a085")
_bold_axis_labels(ax, "Kernel", "Accuracy", "SVM Kernel Comparison (Accuracy)")
plt.xticks(rotation=0)
_save_eps(fig, f"{FIG_DIR}/svm_kernel_accuracy.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 8. SVM HYPERPARAMETER TUNING (GridSearchCV + RandomizedSearchCV)
# --------------------------------------------------------------------------
svm_param_grid = [
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
    {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    {"kernel": ["sigmoid"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    {"kernel": ["poly"], "C": [0.1, 1, 10], "gamma": ["scale"],
     "degree": [2, 3]},
]

print("[SVM] Starting GridSearchCV...", flush=True)
t0 = time.perf_counter()
svm_grid = GridSearchCV(SVC(probability=False, random_state=RANDOM_STATE), svm_param_grid,
                         cv=cv_strategy, scoring="accuracy", n_jobs=1)
svm_grid.fit(X_train_scaled, y_train)
svm_grid_time = time.perf_counter() - t0
print(f"[SVM] GridSearchCV done in {svm_grid_time:.1f}s", flush=True)

print("[SVM] Starting RandomizedSearchCV...", flush=True)
t0 = time.perf_counter()
svm_random = RandomizedSearchCV(SVC(probability=False, random_state=RANDOM_STATE), svm_param_grid,
                                 cv=cv_strategy, scoring="accuracy", n_iter=15,
                                 random_state=RANDOM_STATE, n_jobs=1)
svm_random.fit(X_train_scaled, y_train)
svm_random_time = time.perf_counter() - t0
print(f"[SVM] RandomizedSearchCV done in {svm_random_time:.1f}s", flush=True)

svm_tuning_summary = pd.DataFrame({
    "GridSearchCV": {
        "Best Parameters": str(svm_grid.best_params_),
        "Best CV Accuracy": svm_grid.best_score_,
        "Execution Time (s)": svm_grid_time,
    },
    "RandomizedSearchCV": {
        "Best Parameters": str(svm_random.best_params_),
        "Best CV Accuracy": svm_random.best_score_,
        "Execution Time (s)": svm_random_time,
    },
})
svm_tuning_summary.to_csv(f"{RES_DIR}/svm_tuning_summary.csv")
print("\n=== SVM Tuning Summary ===")
print(svm_tuning_summary)

best_svm_params = svm_grid.best_params_
best_svm = SVC(probability=True, random_state=RANDOM_STATE, **best_svm_params)
y_pred_svm, svm_train_t, svm_pred_t = time_fit_predict(best_svm, X_train_scaled,
                                                         X_test_scaled, y_train)
y_proba_svm = best_svm.predict_proba(X_test_scaled)
svm_metrics = classification_performance_metrics(
    y_test, y_pred_svm, y_proba_svm, model_name="SVM (Tuned)",
    return_dict=True, plot=True, save_path=f"{FIG_DIR}/cm_svm.eps"
)
svm_metrics["Training Time (s)"] = svm_train_t
print("\n=== Tuned SVM Performance ===")
print(svm_metrics)

# --------------------------------------------------------------------------
# 9. HYPERPARAMETER TUNING RESULTS TABLE (combined)
# --------------------------------------------------------------------------
tuning_results_combined = pd.DataFrame({
    "Logistic Regression": {
        "Search Method": "Grid / Random",
        "Best Parameters": str(logreg_grid.best_params_),
        "Best CV Accuracy": logreg_grid.best_score_,
    },
    "SVM": {
        "Search Method": "Grid / Random",
        "Best Parameters": str(svm_grid.best_params_),
        "Best CV Accuracy": svm_grid.best_score_,
    },
}).T
tuning_results_combined.to_csv(f"{RES_DIR}/hyperparameter_tuning_results.csv")
print("\n=== Combined Hyperparameter Tuning Results ===")
print(tuning_results_combined)

# --------------------------------------------------------------------------
# 10. 5-FOLD CROSS-VALIDATION (Logistic Regression vs SVM, best tuned configs)
# --------------------------------------------------------------------------
X_scaled_full = scaler.fit_transform(X)

cv_lr = cross_validate(best_logreg, X_scaled_full, y, cv=cv_strategy, scoring="accuracy")
cv_svm = cross_validate(best_svm, X_scaled_full, y, cv=cv_strategy, scoring="accuracy")

cv_table = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(5)] + ["Average"],
    "Logistic Regression": list(cv_lr["test_score"]) + [cv_lr["test_score"].mean()],
    "SVM": list(cv_svm["test_score"]) + [cv_svm["test_score"].mean()],
}).set_index("Fold")
cv_table.to_csv(f"{RES_DIR}/cross_validation_results.csv")
print("\n=== 5-Fold Cross-Validation Results ===")
print(cv_table)

fig, ax = plt.subplots(figsize=(7, 5))
folds = np.arange(1, 6)
ax.plot(folds, cv_lr["test_score"], marker="o", label="Logistic Regression", color="#2980b9")
ax.plot(folds, cv_svm["test_score"], marker="s", label="SVM", color="#c0392b")
ax.legend()
_bold_axis_labels(ax, "Fold", "Accuracy", "Cross-Validation Accuracy")
_save_eps(fig, f"{FIG_DIR}/cv_accuracy.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 11. ROC CURVES AND COMPARATIVE ANALYSIS
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in [("Logistic Regression (Tuned)", best_logreg), ("SVM (Tuned)", best_svm)]:
    proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.8)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
ax.legend()
_bold_axis_labels(ax, "False Positive Rate", "True Positive Rate", "ROC Curves")
_save_eps(fig, f"{FIG_DIR}/roc_curves.eps")
plt.close(fig)

comparison_summary = pd.DataFrame([logreg_metrics, svm_metrics]).set_index("Model")
comparison_summary.to_csv(f"{RES_DIR}/comparative_analysis.csv")
print("\n=== Comparative Analysis ===")
print(comparison_summary)

fig, ax = plt.subplots(figsize=(8, 5))
comparison_summary[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar", ax=ax)
ax.legend(prop=fm.FontProperties(family="Times New Roman", size=11))
_bold_axis_labels(ax, "Model", "Score", "Logistic Regression vs SVM: Metric Comparison")
plt.xticks(rotation=15, ha="right")
_save_eps(fig, f"{FIG_DIR}/model_comparison_bar.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 12. REGULARIZATION EFFECT (Logistic Regression: accuracy vs C)
# --------------------------------------------------------------------------
C_values = [0.01, 0.1, 1, 10, 100]
reg_effect_rows = []
for C in C_values:
    for penalty in ["l1", "l2"]:
        model = LogisticRegression(C=C, penalty=penalty, solver="liblinear", max_iter=3000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        n_nonzero = int(np.sum(model.coef_[0] != 0))
        reg_effect_rows.append({
            "C": C, "Penalty": penalty,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Non-zero Coefficients": n_nonzero,
        })
reg_effect_df = pd.DataFrame(reg_effect_rows)
reg_effect_df.to_csv(f"{RES_DIR}/regularization_effect.csv", index=False)
print("\n=== Regularization Effect (Logistic Regression) ===")
print(reg_effect_df)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for penalty, color in [("l1", "#8e44ad"), ("l2", "#2980b9")]:
    subset = reg_effect_df[reg_effect_df["Penalty"] == penalty]
    axes[0].plot(subset["C"], subset["Accuracy"], marker="o", label=penalty.upper(), color=color)
    axes[1].plot(subset["C"], subset["Non-zero Coefficients"], marker="o",
                 label=penalty.upper(), color=color)
axes[0].set_xscale("log")
axes[1].set_xscale("log")
axes[0].legend()
axes[1].legend()
_bold_axis_labels(axes[0], "C (log scale)", "Accuracy", "Accuracy vs C")
_bold_axis_labels(axes[1], "C (log scale)", "Non-zero Coefficients", "Sparsity vs C")
plt.tight_layout()
_save_eps(fig, f"{FIG_DIR}/regularization_effect.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 13. SVM DECISION BOUNDARY VISUALIZATION (2D PCA projection, all 4 kernels)
# --------------------------------------------------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_2d = pca.fit_transform(X_train_scaled)

xx, yy = np.meshgrid(np.linspace(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 200),
                      np.linspace(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 200))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for ax, kernel in zip(axes, kernels):
    svm_2d = SVC(kernel=kernel, random_state=RANDOM_STATE)
    svm_2d.fit(X_train_2d, y_train)
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap="coolwarm",
               s=8, edgecolors="k", linewidths=0.2)
    _bold_axis_labels(ax, "PC1", "PC2", f"{kernel.capitalize()} Kernel (2D PCA)")
plt.tight_layout()
_save_eps(fig, f"{FIG_DIR}/svm_decision_boundaries_pca.eps")
plt.close(fig)

print("\nAll figures saved under:", os.path.abspath(FIG_DIR))
print("All result tables saved under:", os.path.abspath(RES_DIR))
print("\nDone.")
