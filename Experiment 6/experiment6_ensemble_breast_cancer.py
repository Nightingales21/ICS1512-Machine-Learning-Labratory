"""
experiment6_ensemble_breast_cancer.py
========================================
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 6 (Lab Manual "Experiment 7"): Bagging, Boosting, and Stacked
Ensemble Models

Uses the reusable module ml_lab_utils.py (from Experiment 1) for:
    - EDA                         -> generate_eda_summary()
    - Classification train/eval   -> train_evaluate_classification()
    - Classification metrics      -> classification_performance_metrics()
    - Global plot style           -> set_plot_style()

Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), 569 samples, 30 numeric
features, binary target (Malignant / Benign). Loaded via
sklearn.datasets.load_breast_cancer, identical to Experiment 5.

NOTE on label encoding: sklearn's load_breast_cancer encodes
target = 0 -> malignant, target = 1 -> benign.
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
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
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})

print("Dataset shape:", df.shape)
print(df["diagnosis"].value_counts())
print("Missing values:", int(df.isnull().sum().sum()))

# --------------------------------------------------------------------------
# 2. EDA (reusable function from Experiment 1)
# --------------------------------------------------------------------------
eda_df = df.drop(columns=["target"])
generate_eda_summary(
    eda_df, target_col="diagnosis", dataset_name="Wisconsin Breast Cancer (Exp. 6)",
    save_path=f"{FIG_DIR}/eda_breast_cancer.eps"
)
plt.close("all")

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
# 4. BAGGING CLASSIFIER (base estimator: Decision Tree)
# --------------------------------------------------------------------------
baseline_models = {
    "Bagging (Baseline)": BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        random_state=RANDOM_STATE)
}
baseline_results_df, baseline_fitted = train_evaluate_classification(
    baseline_models, X_train, X_test, y_train, y_test, scale=False
)
print("\n=== Baseline Bagging (default hyperparameters) ===")
print(baseline_results_df)

bagging_param_grid = {
    "n_estimators": [10, 50, 100],
    "max_samples": [0.5, 0.7, 1.0],
    "max_features": [0.5, 0.7, 1.0],
}

print("\n[Bagging] Starting GridSearchCV (5-fold)...")
t0 = time.perf_counter()
bagging_grid = GridSearchCV(
    BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                       random_state=RANDOM_STATE),
    bagging_param_grid, cv=cv_strategy, scoring="accuracy", n_jobs=1)
bagging_grid.fit(X_train, y_train)
bagging_grid_time = time.perf_counter() - t0
print(f"[Bagging] GridSearchCV done in {bagging_grid_time:.1f}s")
print("Best params:", bagging_grid.best_params_)
print("Best CV accuracy:", bagging_grid.best_score_)

# Table 1: n_estimators x max_samples summary (best over max_features)
bagging_cv_results = pd.DataFrame(bagging_grid.cv_results_)
bagging_table1_rows = []
for n_est in [10, 50, 100]:
    for max_samp in [0.5, 0.7, 1.0]:
        mask = ((bagging_cv_results["param_n_estimators"] == n_est) &
                 (bagging_cv_results["param_max_samples"] == max_samp))
        subset = bagging_cv_results[mask]
        if subset.empty:
            continue
        best_row = subset.loc[subset["mean_test_score"].idxmax()]
        params = best_row["params"]
        model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                                   random_state=RANDOM_STATE, **params)
        cv_res = cross_validate(model, X_train, y_train, cv=cv_strategy,
                                 scoring=["accuracy", "f1"])
        bagging_table1_rows.append({
            "n_estimators": n_est,
            "max_samples": max_samp,
            "Avg CV Accuracy (%)": cv_res["test_accuracy"].mean() * 100,
            "Avg CV F1 Score": cv_res["test_f1"].mean(),
        })
bagging_table1_df = pd.DataFrame(bagging_table1_rows)
bagging_table1_df.to_csv(f"{RES_DIR}/bagging_hyperparameter_evaluation.csv", index=False)
print("\n=== Table 1: Bagging Hyperparameter Evaluation (5-Fold CV) ===")
print(bagging_table1_df)

best_bagging = bagging_grid.best_estimator_
y_pred_bag, bag_train_t, bag_pred_t = time_fit_predict(best_bagging, X_train, X_test, y_train)
y_proba_bag = best_bagging.predict_proba(X_test)
bagging_metrics = classification_performance_metrics(
    y_test, y_pred_bag, y_proba_bag, model_name="Bagging (Tuned)",
    return_dict=True, plot=True, save_path=f"{FIG_DIR}/cm_bagging.eps"
)
bagging_metrics["Training Time (s)"] = bag_train_t
print("\n=== Tuned Bagging Performance ===")
print(bagging_metrics)

# --------------------------------------------------------------------------
# 5. BOOSTING CLASSIFIERS (AdaBoost and Gradient Boosting)
# --------------------------------------------------------------------------
boosting_baseline = {
    "AdaBoost (Baseline)": AdaBoostClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting (Baseline)": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
boosting_baseline_df, boosting_baseline_fitted = train_evaluate_classification(
    boosting_baseline, X_train, X_test, y_train, y_test, scale=False
)
print("\n=== Baseline Boosting Models (default hyperparameters) ===")
print(boosting_baseline_df)

# AdaBoost hyperparameter tuning
ada_param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 1.0],
}
print("\n[AdaBoost] Starting GridSearchCV (5-fold)...")
t0 = time.perf_counter()
ada_grid = GridSearchCV(AdaBoostClassifier(random_state=RANDOM_STATE), ada_param_grid,
                         cv=cv_strategy, scoring="accuracy", n_jobs=1)
ada_grid.fit(X_train, y_train)
ada_grid_time = time.perf_counter() - t0
print(f"[AdaBoost] GridSearchCV done in {ada_grid_time:.1f}s")
print("Best params:", ada_grid.best_params_)
print("Best CV accuracy:", ada_grid.best_score_)

# Gradient Boosting hyperparameter tuning
gb_param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 1.0],
    "max_depth": [2, 3, 5],
}
print("\n[Gradient Boosting] Starting GridSearchCV (5-fold)...")
t0 = time.perf_counter()
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=RANDOM_STATE), gb_param_grid,
                        cv=cv_strategy, scoring="accuracy", n_jobs=1)
gb_grid.fit(X_train, y_train)
gb_grid_time = time.perf_counter() - t0
print(f"[Gradient Boosting] GridSearchCV done in {gb_grid_time:.1f}s")
print("Best params:", gb_grid.best_params_)
print("Best CV accuracy:", gb_grid.best_score_)

# Table 2: n_estimators x learning_rate summary, using AdaBoost as the
# representative boosting algorithm for this table shape (max_depth held at
# GB's own grid separately, reported alongside)
ada_cv_results = pd.DataFrame(ada_grid.cv_results_)
boosting_table2_rows = []
for n_est in [50, 100, 200]:
    for lr in [0.01, 0.1, 1.0]:
        mask = ((ada_cv_results["param_n_estimators"] == n_est) &
                 (ada_cv_results["param_learning_rate"] == lr))
        subset = ada_cv_results[mask]
        if subset.empty:
            continue
        params = subset.iloc[0]["params"]
        model = AdaBoostClassifier(random_state=RANDOM_STATE, **params)
        cv_res = cross_validate(model, X_train, y_train, cv=cv_strategy,
                                 scoring=["accuracy", "f1"])
        boosting_table2_rows.append({
            "Algorithm": "AdaBoost",
            "n_estimators": n_est,
            "learning_rate": lr,
            "Avg CV Accuracy (%)": cv_res["test_accuracy"].mean() * 100,
            "Avg CV F1 Score": cv_res["test_f1"].mean(),
        })
boosting_table2_df = pd.DataFrame(boosting_table2_rows)
boosting_table2_df.to_csv(f"{RES_DIR}/boosting_hyperparameter_evaluation.csv", index=False)
print("\n=== Table 2: Boosting (AdaBoost) Hyperparameter Evaluation (5-Fold CV) ===")
print(boosting_table2_df)

# Choose the better of tuned AdaBoost vs tuned Gradient Boosting as "Boosting"
best_ada = ada_grid.best_estimator_
best_gb = gb_grid.best_estimator_
if ada_grid.best_score_ >= gb_grid.best_score_:
    best_boosting = best_ada
    best_boosting_name = "AdaBoost (Tuned)"
else:
    best_boosting = best_gb
    best_boosting_name = "Gradient Boosting (Tuned)"
print(f"\nSelected as overall 'Boosting' model: {best_boosting_name}")

y_pred_boost, boost_train_t, boost_pred_t = time_fit_predict(best_boosting, X_train, X_test, y_train)
y_proba_boost = best_boosting.predict_proba(X_test)
boosting_metrics = classification_performance_metrics(
    y_test, y_pred_boost, y_proba_boost, model_name="Boosting (Tuned)",
    return_dict=True, plot=True, save_path=f"{FIG_DIR}/cm_boosting.eps"
)
boosting_metrics["Training Time (s)"] = boost_train_t
boosting_metrics["Selected Algorithm"] = best_boosting_name
print("\n=== Tuned Boosting Performance ===")
print(boosting_metrics)

# --------------------------------------------------------------------------
# 6. STACKED ENSEMBLE (Base: SVM, Naive Bayes, Decision Tree; Meta: LogReg)
# --------------------------------------------------------------------------
base_learners = [
    ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
    ("nb", GaussianNB()),
    ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
]
stacking_baseline = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=cv_strategy
)
stacking_models = {"Stacked Ensemble (Baseline)": stacking_baseline}
stacking_baseline_df, stacking_baseline_fitted = train_evaluate_classification(
    stacking_models, X_train, X_test, y_train, y_test, scale=False
)
print("\n=== Baseline Stacked Ensemble ===")
print(stacking_baseline_df)

# Compare a couple of base-model / meta-learner combinations (Table 3)
stacking_configs = {
    "SVM+NB+DT / LogReg": StackingClassifier(
        estimators=[("svm", SVC(probability=True, random_state=RANDOM_STATE)),
                    ("nb", GaussianNB()),
                    ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))],
        final_estimator=LogisticRegression(max_iter=2000), cv=cv_strategy),
    "SVM+NB+DT / DecisionTree": StackingClassifier(
        estimators=[("svm", SVC(probability=True, random_state=RANDOM_STATE)),
                    ("nb", GaussianNB()),
                    ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))],
        final_estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
        cv=cv_strategy),
    "NB+DT / LogReg": StackingClassifier(
        estimators=[("nb", GaussianNB()),
                    ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))],
        final_estimator=LogisticRegression(max_iter=2000), cv=cv_strategy),
}

print("\n[Stacking] Evaluating base-model / meta-learner combinations (5-fold CV)...")
stacking_table3_rows = []
for combo_name, model in stacking_configs.items():
    t0 = time.perf_counter()
    cv_res = cross_validate(model, X_train, y_train, cv=cv_strategy, scoring=["accuracy", "f1"])
    combo_time = time.perf_counter() - t0
    base_str, meta_str = combo_name.split(" / ")
    stacking_table3_rows.append({
        "Base Models": base_str,
        "Meta Learner": meta_str,
        "Avg CV Accuracy (%)": cv_res["test_accuracy"].mean() * 100,
        "Avg CV F1 Score": cv_res["test_f1"].mean(),
    })
    print(f"  {combo_name}: CV Accuracy = {cv_res['test_accuracy'].mean()*100:.2f}%, "
          f"time = {combo_time:.1f}s")

stacking_table3_df = pd.DataFrame(stacking_table3_rows)
stacking_table3_df.to_csv(f"{RES_DIR}/stacking_hyperparameter_evaluation.csv", index=False)
print("\n=== Table 3: Stacked Ensemble Evaluation (5-Fold CV) ===")
print(stacking_table3_df)

best_combo_idx = stacking_table3_df["Avg CV Accuracy (%)"].idxmax()
best_combo_name = list(stacking_configs.keys())[best_combo_idx]
best_stacking = stacking_configs[best_combo_name]
print(f"\nBest stacking configuration: {best_combo_name}")

y_pred_stack, stack_train_t, stack_pred_t = time_fit_predict(best_stacking, X_train, X_test, y_train)
y_proba_stack = best_stacking.predict_proba(X_test)
stacking_metrics = classification_performance_metrics(
    y_test, y_pred_stack, y_proba_stack, model_name="Stacked Ensemble (Tuned)",
    return_dict=True, plot=True, save_path=f"{FIG_DIR}/cm_stacking.eps"
)
stacking_metrics["Training Time (s)"] = stack_train_t
stacking_metrics["Best Configuration"] = best_combo_name
print("\n=== Tuned Stacked Ensemble Performance ===")
print(stacking_metrics)

# --------------------------------------------------------------------------
# 7. HYPERPARAMETER TUNING RESULTS SUMMARY
# --------------------------------------------------------------------------
tuning_summary = pd.DataFrame({
    "Bagging": {
        "Search Method": "GridSearchCV (5-fold)",
        "Best Parameters": str(bagging_grid.best_params_),
        "Best CV Accuracy": bagging_grid.best_score_,
    },
    "AdaBoost": {
        "Search Method": "GridSearchCV (5-fold)",
        "Best Parameters": str(ada_grid.best_params_),
        "Best CV Accuracy": ada_grid.best_score_,
    },
    "Gradient Boosting": {
        "Search Method": "GridSearchCV (5-fold)",
        "Best Parameters": str(gb_grid.best_params_),
        "Best CV Accuracy": gb_grid.best_score_,
    },
    "Stacked Ensemble": {
        "Search Method": "Manual combo search (5-fold CV)",
        "Best Parameters": best_combo_name,
        "Best CV Accuracy": stacking_table3_df["Avg CV Accuracy (%)"].max() / 100,
    },
}).T
tuning_summary.to_csv(f"{RES_DIR}/hyperparameter_tuning_results.csv")
print("\n=== Hyperparameter Tuning Results Summary ===")
print(tuning_summary)

# --------------------------------------------------------------------------
# 8. PERFORMANCE COMPARISON TABLE (Table 4)
# --------------------------------------------------------------------------
performance_comparison = pd.DataFrame([bagging_metrics, boosting_metrics, stacking_metrics])
performance_comparison = performance_comparison[
    ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Training Time (s)"]
].set_index("Model")
performance_comparison.to_csv(f"{RES_DIR}/performance_comparison.csv")
print("\n=== Table 4: Performance Comparison of Ensemble Models ===")
print(performance_comparison)

fig, ax = plt.subplots(figsize=(9, 5))
performance_comparison[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar", ax=ax)
ax.legend(prop=fm.FontProperties(family="Times New Roman", size=11))
_bold_axis_labels(ax, "Model", "Score", "Ensemble Model Comparison")
plt.xticks(rotation=15, ha="right")
_save_eps(fig, f"{FIG_DIR}/ensemble_comparison_bar.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 9. 5-FOLD CROSS-VALIDATION COMPARISON (all three tuned ensembles)
# --------------------------------------------------------------------------
cv_bag = cross_validate(best_bagging, X_train, y_train, cv=cv_strategy, scoring="accuracy")
cv_boost = cross_validate(best_boosting, X_train, y_train, cv=cv_strategy, scoring="accuracy")
cv_stack = cross_validate(best_stacking, X_train, y_train, cv=cv_strategy, scoring="accuracy")

cv_table = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(5)] + ["Average"],
    "Bagging": list(cv_bag["test_score"]) + [cv_bag["test_score"].mean()],
    "Boosting": list(cv_boost["test_score"]) + [cv_boost["test_score"].mean()],
    "Stacked Ensemble": list(cv_stack["test_score"]) + [cv_stack["test_score"].mean()],
}).set_index("Fold")
cv_table.to_csv(f"{RES_DIR}/cv_accuracy_comparison.csv")
print("\n=== 5-Fold Cross-Validation Comparison ===")
print(cv_table)

fig, ax = plt.subplots(figsize=(8, 5))
folds = np.arange(1, 6)
ax.plot(folds, cv_bag["test_score"], marker="o", label="Bagging", color="#2980b9")
ax.plot(folds, cv_boost["test_score"], marker="s", label="Boosting", color="#c0392b")
ax.plot(folds, cv_stack["test_score"], marker="^", label="Stacked Ensemble", color="#16a085")
ax.legend()
_bold_axis_labels(ax, "Fold", "Accuracy", "5-Fold Cross-Validation Accuracy Comparison")
_save_eps(fig, f"{FIG_DIR}/cv_accuracy_comparison.eps")
plt.close(fig)

# Fold-to-fold standard deviation as a stability proxy
stability_df = pd.DataFrame({
    "Model": ["Bagging", "Boosting", "Stacked Ensemble"],
    "CV Mean Accuracy": [cv_bag["test_score"].mean(), cv_boost["test_score"].mean(),
                          cv_stack["test_score"].mean()],
    "CV Std Dev (Stability)": [cv_bag["test_score"].std(), cv_boost["test_score"].std(),
                                cv_stack["test_score"].std()],
}).set_index("Model")
stability_df.to_csv(f"{RES_DIR}/stability_comparison.csv")
print("\n=== Stability Comparison (CV std dev) ===")
print(stability_df)

# --------------------------------------------------------------------------
# 10. ROC CURVES (all three tuned ensembles)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in [("Bagging (Tuned)", best_bagging),
                     (f"Boosting (Tuned - {best_boosting_name.split(' ')[0]})", best_boosting),
                     ("Stacked Ensemble (Tuned)", best_stacking)]:
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.8)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
ax.legend(fontsize=9)
_bold_axis_labels(ax, "False Positive Rate", "True Positive Rate", "ROC Curves")
_save_eps(fig, f"{FIG_DIR}/roc_curves.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 11. BIAS-VARIANCE ANALYSIS: single tree vs bagging vs boosting (train/val gap)
# --------------------------------------------------------------------------
bias_variance_rows = []
single_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
for name, model in [("Single Decision Tree", single_tree),
                     ("Bagging (Tuned)", best_bagging),
                     ("Boosting (Tuned)", best_boosting),
                     ("Stacked Ensemble (Tuned)", best_stacking)]:
    cv_res = cross_validate(model, X_train, y_train, cv=cv_strategy,
                             scoring="accuracy", return_train_score=True)
    bias_variance_rows.append({
        "Model": name,
        "Train Accuracy": cv_res["train_score"].mean(),
        "Validation Accuracy": cv_res["test_score"].mean(),
        "Train-Val Gap": cv_res["train_score"].mean() - cv_res["test_score"].mean(),
    })
bias_variance_df = pd.DataFrame(bias_variance_rows).set_index("Model")
bias_variance_df.to_csv(f"{RES_DIR}/bias_variance_analysis.csv")
print("\n=== Bias-Variance Analysis (Train-Validation Gap) ===")
print(bias_variance_df)

fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(len(bias_variance_df))
width = 0.35
ax.bar(x_pos - width/2, bias_variance_df["Train Accuracy"], width,
       label="Training Accuracy", color="#2980b9")
ax.bar(x_pos + width/2, bias_variance_df["Validation Accuracy"], width,
       label="Validation Accuracy", color="#c0392b")
ax.set_xticks(x_pos)
ax.set_xticklabels(bias_variance_df.index, rotation=15, ha="right")
ax.legend()
_bold_axis_labels(ax, "Model", "Accuracy", "Bias-Variance: Training vs Validation Accuracy")
_save_eps(fig, f"{FIG_DIR}/bias_variance_analysis.eps")
plt.close(fig)

print("\nAll figures saved under:", os.path.abspath(FIG_DIR))
print("All result tables saved under:", os.path.abspath(RES_DIR))
print("\nDone.")
