"""
experiment5_dt_rf_breast_cancer.py
====================================
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 5: Decision Tree and Random Forest - A Comparative Classification Study

Uses the reusable module ml_lab_utils.py (from Experiment 1) for:
    - EDA                         -> generate_eda_summary()
    - Classification train/eval   -> train_evaluate_classification()
    - Classification metrics      -> classification_performance_metrics()
    - Global plot style           -> set_plot_style()

Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), 569 samples, 30 numeric
features, binary target (Malignant / Benign). Loaded via
sklearn.datasets.load_breast_cancer (identical data to the UCI/Kaggle
Wisconsin Diagnostic Breast Cancer dataset specified in the lab manual).

NOTE on label encoding: sklearn's load_breast_cancer encodes
target = 0 -> malignant, target = 1 -> benign (i.e. "0" is the more
serious/positive-for-disease class). This is the reverse of the
0=negative/1=positive convention many students expect, so it is called out
explicitly here and in the report to avoid misreading the confusion matrix.
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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_validate, GridSearchCV)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
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
# 1. LOAD DATASET AND ENCODE CLASS LABELS
# --------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
# target: 0 = malignant, 1 = benign (sklearn's native encoding)
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})

print("Dataset shape:", df.shape)
print(df["diagnosis"].value_counts())
print("Missing values:", int(df.isnull().sum().sum()))

# --------------------------------------------------------------------------
# 2. EDA (reusable function from Experiment 1)
# --------------------------------------------------------------------------
eda_df = df.drop(columns=["target"])
generate_eda_summary(
    eda_df, target_col="diagnosis", dataset_name="Wisconsin Breast Cancer",
    save_path=f"{FIG_DIR}/eda_breast_cancer.eps"
)
plt.close("all")

# Feature correlation heatmap (full 30x30, required explicitly by manual)
fig, ax = plt.subplots(figsize=(14, 12))
corr = df[data.feature_names].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar=True,
            xticklabels=True, yticklabels=True)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7, rotation=0)
_bold_axis_labels(ax, title="Full Feature Correlation Heatmap (30 features)")
_save_eps(fig, f"{FIG_DIR}/full_correlation_heatmap.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (80-20, stratified)
# --------------------------------------------------------------------------
X = df[list(data.feature_names)].values
y = df["target"].values  # 0 = malignant, 1 = benign
feature_names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

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
# 4. BASELINE DECISION TREE (via reusable train_evaluate_classification)
# --------------------------------------------------------------------------
baseline_models = {"Decision Tree (Baseline)": DecisionTreeClassifier(random_state=RANDOM_STATE)}
baseline_results_df, baseline_fitted = train_evaluate_classification(
    baseline_models, X_train, X_test, y_train, y_test, scale=False
)
print("\n=== Baseline Decision Tree (default hyperparameters) ===")
print(baseline_results_df)

# --------------------------------------------------------------------------
# 5. DECISION TREE HYPERPARAMETER SEARCH SPACE + 5-FOLD CV EVALUATION
# --------------------------------------------------------------------------
dt_param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

print("\n[Decision Tree] Starting GridSearchCV (5-fold)...")
t0 = time.perf_counter()
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), dt_param_grid,
                        cv=cv_strategy, scoring="accuracy", n_jobs=1)
dt_grid.fit(X_train, y_train)
dt_grid_time = time.perf_counter() - t0
print(f"[Decision Tree] GridSearchCV done in {dt_grid_time:.1f}s")
print("Best params:", dt_grid.best_params_)
print("Best CV accuracy:", dt_grid.best_score_)

# Table 1: Criterion x Max Depth summary (averaging over the other params via
# best-per-combination selection, as requested by the manual's table shape)
dt_cv_results = pd.DataFrame(dt_grid.cv_results_)
dt_table1_rows = []
for criterion in ["gini", "entropy"]:
    for max_depth in [3, 5, 7, 10, None]:
        if max_depth is None:
            mask = dt_cv_results["param_max_depth"].isna()
        else:
            mask = dt_cv_results["param_max_depth"] == max_depth
        subset = dt_cv_results[(dt_cv_results["param_criterion"] == criterion) & mask]
        if subset.empty:
            continue
        best_row = subset.loc[subset["mean_test_score"].idxmax()]
        # Compute F1 for this specific best combination via a fresh 5-fold CV
        params = best_row["params"]
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
        cv_res = cross_validate(model, X_train, y_train, cv=cv_strategy,
                                 scoring=["accuracy", "f1"])
        dt_table1_rows.append({
            "Criterion": criterion,
            "Max Depth": "None" if max_depth is None else max_depth,
            "Avg CV Accuracy (%)": cv_res["test_accuracy"].mean() * 100,
            "Avg CV F1 Score": cv_res["test_f1"].mean(),
        })
dt_table1_df = pd.DataFrame(dt_table1_rows)
dt_table1_df.to_csv(f"{RES_DIR}/dt_hyperparameter_evaluation.csv", index=False)
print("\n=== Table 1: Decision Tree Hyperparameter Evaluation (5-Fold CV) ===")
print(dt_table1_df)

best_dt = dt_grid.best_estimator_
y_pred_dt, dt_train_t, dt_pred_t = time_fit_predict(best_dt, X_train, X_test, y_train)
y_proba_dt = best_dt.predict_proba(X_test)
dt_metrics = classification_performance_metrics(
    y_test, y_pred_dt, y_proba_dt, model_name="Decision Tree (Tuned)",
    return_dict=True, plot=True, save_path=f"{FIG_DIR}/cm_decision_tree.eps"
)
dt_metrics["Training Time (s)"] = dt_train_t
print("\n=== Tuned Decision Tree Performance ===")
print(dt_metrics)

# --------------------------------------------------------------------------
# 6. OVERFITTING ANALYSIS: TREE DEPTH vs TRAIN/VALIDATION ACCURACY
# --------------------------------------------------------------------------
depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None]
depth_rows = []
for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    cv_res = cross_validate(model, X_train, y_train, cv=cv_strategy,
                             scoring="accuracy", return_train_score=True)
    depth_rows.append({
        "Max Depth": "None" if d is None else d,
        "Train Accuracy": cv_res["train_score"].mean(),
        "Validation Accuracy": cv_res["test_score"].mean(),
    })
depth_df = pd.DataFrame(depth_rows)
depth_df.to_csv(f"{RES_DIR}/depth_vs_accuracy.csv", index=False)
print("\n=== Tree Depth vs Train/Validation Accuracy ===")
print(depth_df)

fig, ax = plt.subplots(figsize=(8, 5))
x_labels = [str(d) for d in depth_df["Max Depth"]]
x_pos = np.arange(len(x_labels))
ax.plot(x_pos, depth_df["Train Accuracy"], marker="o", label="Training Accuracy", color="#2980b9")
ax.plot(x_pos, depth_df["Validation Accuracy"], marker="s", label="Validation Accuracy", color="#c0392b")
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.legend()
_bold_axis_labels(ax, "Max Depth", "Accuracy", "Decision Tree: Overfitting vs Tree Depth")
_save_eps(fig, f"{FIG_DIR}/depth_vs_accuracy.eps")
plt.close(fig)

# Visualize the tuned (best) decision tree structure (top levels only, for readability)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(best_dt, max_depth=3, feature_names=feature_names,
          class_names=["Malignant", "Benign"], filled=True, fontsize=8, ax=ax)
ax.set_title("Best Decision Tree Structure (first 3 levels shown)",
             fontproperties=fm.FontProperties(family="Times New Roman", weight="bold", size=15))
_save_eps(fig, f"{FIG_DIR}/decision_tree_structure.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 7. RANDOM FOREST HYPERPARAMETER SEARCH SPACE + 5-FOLD CV EVALUATION
# --------------------------------------------------------------------------
rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False],
}

print("\n[Random Forest] Starting GridSearchCV (5-fold)...")
t0 = time.perf_counter()
rf_grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), rf_param_grid,
                        cv=cv_strategy, scoring="accuracy", n_jobs=1)
rf_grid.fit(X_train, y_train)
rf_grid_time = time.perf_counter() - t0
print(f"[Random Forest] GridSearchCV done in {rf_grid_time:.1f}s")
print("Best params:", rf_grid.best_params_)
print("Best CV accuracy:", rf_grid.best_score_)

# Table 2: n_estimators x Max Depth summary (best over max_features/bootstrap)
rf_cv_results = pd.DataFrame(rf_grid.cv_results_)
rf_table2_rows = []
for n_est in [50, 100, 200]:
    for max_depth in [5, 10, None]:
        if max_depth is None:
            depth_mask = rf_cv_results["param_max_depth"].isna()
        else:
            depth_mask = rf_cv_results["param_max_depth"] == max_depth
        subset = rf_cv_results[(rf_cv_results["param_n_estimators"] == n_est) & depth_mask]
        if subset.empty:
            continue
        best_row = subset.loc[subset["mean_test_score"].idxmax()]
        params = best_row["params"]
        model = RandomForestClassifier(random_state=RANDOM_STATE, **params)
        cv_res = cross_validate(model, X_train, y_train, cv=cv_strategy,
                                 scoring=["accuracy", "f1"])
        rf_table2_rows.append({
            "n_estimators": n_est,
            "Max Depth": "None" if max_depth is None else max_depth,
            "Max Features": params["max_features"],
            "Avg CV Accuracy (%)": cv_res["test_accuracy"].mean() * 100,
            "Avg CV F1 Score": cv_res["test_f1"].mean(),
        })
rf_table2_df = pd.DataFrame(rf_table2_rows)
rf_table2_df.to_csv(f"{RES_DIR}/rf_hyperparameter_evaluation.csv", index=False)
print("\n=== Table 2: Random Forest Hyperparameter Evaluation (5-Fold CV) ===")
print(rf_table2_df)

best_rf = rf_grid.best_estimator_
y_pred_rf, rf_train_t, rf_pred_t = time_fit_predict(best_rf, X_train, X_test, y_train)
y_proba_rf = best_rf.predict_proba(X_test)
rf_metrics = classification_performance_metrics(
    y_test, y_pred_rf, y_proba_rf, model_name="Random Forest (Tuned)",
    return_dict=True, plot=True, save_path=f"{FIG_DIR}/cm_random_forest.eps"
)
rf_metrics["Training Time (s)"] = rf_train_t
print("\n=== Tuned Random Forest Performance ===")
print(rf_metrics)

# --------------------------------------------------------------------------
# 8. HYPERPARAMETER TUNING RESULTS SUMMARY TABLE
# --------------------------------------------------------------------------
tuning_summary = pd.DataFrame({
    "Decision Tree": {
        "Search Method": "GridSearchCV (5-fold)",
        "Best Parameters": str(dt_grid.best_params_),
        "Best CV Accuracy": dt_grid.best_score_,
    },
    "Random Forest": {
        "Search Method": "GridSearchCV (5-fold)",
        "Best Parameters": str(rf_grid.best_params_),
        "Best CV Accuracy": rf_grid.best_score_,
    },
}).T
tuning_summary.to_csv(f"{RES_DIR}/hyperparameter_tuning_results.csv")
print("\n=== Hyperparameter Tuning Results Summary ===")
print(tuning_summary)

# --------------------------------------------------------------------------
# 9. 5-FOLD CROSS-VALIDATION PERFORMANCE COMPARISON (Table 3)
# --------------------------------------------------------------------------
cv_dt = cross_validate(best_dt, X_train, y_train, cv=cv_strategy, scoring="accuracy")
cv_rf = cross_validate(best_rf, X_train, y_train, cv=cv_strategy, scoring="accuracy")

cv_table3 = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(5)] + ["Average"],
    "Decision Tree": list(cv_dt["test_score"]) + [cv_dt["test_score"].mean()],
    "Random Forest": list(cv_rf["test_score"]) + [cv_rf["test_score"].mean()],
}).set_index("Fold")
cv_table3.to_csv(f"{RES_DIR}/cv_accuracy_comparison.csv")
print("\n=== Table 3: 5-Fold Cross-Validation Accuracy Comparison ===")
print(cv_table3)

fig, ax = plt.subplots(figsize=(7, 5))
folds = np.arange(1, 6)
ax.plot(folds, cv_dt["test_score"], marker="o", label="Decision Tree", color="#c0392b")
ax.plot(folds, cv_rf["test_score"], marker="s", label="Random Forest", color="#2980b9")
ax.legend()
_bold_axis_labels(ax, "Fold", "Accuracy", "5-Fold Cross-Validation Accuracy Comparison")
_save_eps(fig, f"{FIG_DIR}/cv_accuracy_comparison.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 10. EVALUATION METRICS: ROC CURVES (both models on test set)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in [("Decision Tree (Tuned)", best_dt), ("Random Forest (Tuned)", best_rf)]:
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.8)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
ax.legend()
_bold_axis_labels(ax, "False Positive Rate", "True Positive Rate", "ROC Curves")
_save_eps(fig, f"{FIG_DIR}/roc_curves.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 11. RANDOM FOREST: n_estimators vs ACCURACY (ensemble size effect)
# --------------------------------------------------------------------------
n_est_values = [1, 5, 10, 25, 50, 100, 150, 200, 300]
n_est_rows = []
for n in n_est_values:
    model = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE)
    cv_res = cross_validate(model, X_train, y_train, cv=cv_strategy, scoring="accuracy")
    n_est_rows.append({"n_estimators": n, "CV Accuracy": cv_res["test_score"].mean()})
n_est_df = pd.DataFrame(n_est_rows)
n_est_df.to_csv(f"{RES_DIR}/n_estimators_vs_accuracy.csv", index=False)
print("\n=== Random Forest: n_estimators vs CV Accuracy ===")
print(n_est_df)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(n_est_df["n_estimators"], n_est_df["CV Accuracy"], marker="o", color="#16a085")
_bold_axis_labels(ax, "Number of Trees (n_estimators)", "CV Accuracy",
                   "Random Forest: Ensemble Size vs Accuracy")
_save_eps(fig, f"{FIG_DIR}/n_estimators_vs_accuracy.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 12. FEATURE IMPORTANCE COMPARISON (Decision Tree vs Random Forest)
# --------------------------------------------------------------------------
dt_importance = pd.Series(best_dt.feature_importances_, index=feature_names)
rf_importance = pd.Series(best_rf.feature_importances_, index=feature_names)
importance_df = pd.DataFrame({
    "Decision Tree": dt_importance,
    "Random Forest": rf_importance,
}).sort_values("Random Forest", ascending=False)
importance_df.to_csv(f"{RES_DIR}/feature_importance_comparison.csv")
print("\n=== Feature Importance Comparison (top 10) ===")
print(importance_df.head(10))

top10 = importance_df.head(10)
fig, ax = plt.subplots(figsize=(9, 6))
top10.plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.legend(prop=fm.FontProperties(family="Times New Roman", size=11))
_bold_axis_labels(ax, "Importance", "Feature", "Feature Importance: Decision Tree vs Random Forest (Top 10)")
_save_eps(fig, f"{FIG_DIR}/feature_importance_comparison.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 13. COMPARATIVE SUMMARY TABLE
# --------------------------------------------------------------------------
comparison_summary = pd.DataFrame([dt_metrics, rf_metrics]).set_index("Model")
comparison_summary.to_csv(f"{RES_DIR}/comparative_analysis.csv")
print("\n=== Comparative Analysis ===")
print(comparison_summary)

fig, ax = plt.subplots(figsize=(8, 5))
comparison_summary[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar", ax=ax)
ax.legend(prop=fm.FontProperties(family="Times New Roman", size=11))
_bold_axis_labels(ax, "Model", "Score", "Decision Tree vs Random Forest: Metric Comparison")
plt.xticks(rotation=15, ha="right")
_save_eps(fig, f"{FIG_DIR}/model_comparison_bar.eps")
plt.close(fig)

print("\nAll figures saved under:", os.path.abspath(FIG_DIR))
print("All result tables saved under:", os.path.abspath(RES_DIR))
print("\nDone.")
