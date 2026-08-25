"""
experiment2_spambase.py
========================
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 2: Email Spam/Ham Classification using Naive Bayes and KNN

Uses the reusable module ml_lab_utils.py (from Experiment 1) for:
    - EDA                         -> generate_eda_summary()
    - Classification train/eval   -> train_evaluate_classification()
    - Classification metrics      -> classification_performance_metrics()
    - Global plot style           -> set_plot_style()

Dataset: Spambase (UCI ML Repository / Kaggle mirror), 4601 emails x 57
features + binary target (1 = spam, 0 = ham).
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
from scipy import stats

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      RandomizedSearchCV, StratifiedKFold,
                                      cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              precision_recall_curve, confusion_matrix)

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
# 1. LOAD DATASET
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
# 3. EDA  (reusable function from Experiment 1)
# --------------------------------------------------------------------------
generate_eda_summary(
    df, target_col="spam", dataset_name="Spambase",
    save_path=f"{FIG_DIR}/eda_spambase.eps"
)
plt.close("all")

# --------------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT + FEATURE SCALING
# --------------------------------------------------------------------------
X = df.drop(columns=["spam"]).values
y = df["spam"].values
feature_names = df.drop(columns=["spam"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Non-negative version for MultinomialNB (which requires X >= 0); StandardScaler
# output has negative values, so Multinomial NB uses the raw (unscaled) features.
X_train_raw, X_test_raw = X_train, X_test

# --------------------------------------------------------------------------
# 5. NAIVE BAYES: GAUSSIAN, MULTINOMIAL, BERNOULLI
# --------------------------------------------------------------------------
nb_results = []
nb_models = {}
nb_timings = {}

def time_fit_predict(model, Xtr, Xte, ytr):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = model.predict(Xte)
    pred_t = time.perf_counter() - t0
    return y_pred, train_t, pred_t

# Gaussian NB -> scaled continuous features
gnb = GaussianNB()
y_pred, train_t, pred_t = time_fit_predict(gnb, X_train_scaled, X_test_scaled, y_train)
y_proba = gnb.predict_proba(X_test_scaled)
m = classification_performance_metrics(y_test, y_pred, y_proba, model_name="Gaussian NB",
                                        return_dict=True, plot=True,
                                        save_path=f"{FIG_DIR}/cm_gaussian_nb.eps")
m.update({"Train Time (s)": train_t, "Predict Time (s)": pred_t})
nb_results.append(m)
nb_models["Gaussian NB"] = gnb
nb_timings["Gaussian NB"] = (train_t, pred_t)

# Multinomial NB -> raw non-negative word/char frequency features
mnb = MultinomialNB()
y_pred, train_t, pred_t = time_fit_predict(mnb, X_train_raw, X_test_raw, y_train)
y_proba = mnb.predict_proba(X_test_raw)
m = classification_performance_metrics(y_test, y_pred, y_proba, model_name="Multinomial NB",
                                        return_dict=True, plot=True,
                                        save_path=f"{FIG_DIR}/cm_multinomial_nb.eps")
m.update({"Train Time (s)": train_t, "Predict Time (s)": pred_t})
nb_results.append(m)
nb_models["Multinomial NB"] = mnb
nb_timings["Multinomial NB"] = (train_t, pred_t)

# Bernoulli NB -> binarized presence/absence of features (threshold 0)
bnb = BernoulliNB()
y_pred, train_t, pred_t = time_fit_predict(bnb, X_train_raw, X_test_raw, y_train)
y_proba = bnb.predict_proba(X_test_raw)
m = classification_performance_metrics(y_test, y_pred, y_proba, model_name="Bernoulli NB",
                                        return_dict=True, plot=True,
                                        save_path=f"{FIG_DIR}/cm_bernoulli_nb.eps")
m.update({"Train Time (s)": train_t, "Predict Time (s)": pred_t})
nb_results.append(m)
nb_models["Bernoulli NB"] = bnb
nb_timings["Bernoulli NB"] = (train_t, pred_t)

nb_results_df = pd.DataFrame(nb_results).set_index("Model")
nb_results_df.to_csv(f"{RES_DIR}/naive_bayes_comparison.csv")
print("\n=== Naive Bayes Comparison ===")
print(nb_results_df)

best_nb_name = nb_results_df["Accuracy"].idxmax()
best_nb_model = nb_models[best_nb_name]
best_nb_X_train = X_train_scaled if best_nb_name == "Gaussian NB" else X_train_raw
best_nb_X_test = X_test_scaled if best_nb_name == "Gaussian NB" else X_test_raw
print(f"\nBest Naive Bayes variant: {best_nb_name}")

# --------------------------------------------------------------------------
# 6. KNN: EFFECT OF VARYING k
# --------------------------------------------------------------------------
k_values = [1, 3, 5, 7, 9, 11]
knn_k_results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    knn_k_results.append({
        "k": k,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    })
knn_k_df = pd.DataFrame(knn_k_results).set_index("k")
knn_k_df.to_csv(f"{RES_DIR}/knn_k_comparison.csv")
print("\n=== KNN: Accuracy vs k ===")
print(knn_k_df)

best_k = knn_k_df["Accuracy"].idxmax()
print(f"Best k (plain KNN sweep): {best_k}")

# Plot: Accuracy vs k
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(knn_k_df.index, knn_k_df["Accuracy"], marker="o", color="#2980b9", linewidth=2)
_bold_axis_labels(ax, "k (Number of Neighbors)", "Accuracy", "Accuracy vs k (KNN)")
_save_eps(fig, f"{FIG_DIR}/accuracy_vs_k.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 7. GRIDSEARCHCV vs RANDOMIZEDSEARCHCV FOR KNN
# --------------------------------------------------------------------------
param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
    "algorithm": ["kd_tree", "ball_tree"],
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

t0 = time.perf_counter()
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv_strategy,
                            scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
grid_time = time.perf_counter() - t0

t0 = time.perf_counter()
random_search = RandomizedSearchCV(KNeighborsClassifier(), param_grid, cv=cv_strategy,
                                    scoring="accuracy", n_iter=20,
                                    random_state=RANDOM_STATE, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)
random_time = time.perf_counter() - t0

search_comparison = pd.DataFrame({
    "GridSearchCV": {
        "Best k": grid_search.best_params_["n_neighbors"],
        "Metric": grid_search.best_params_["metric"],
        "Weights": grid_search.best_params_["weights"],
        "Algorithm": grid_search.best_params_["algorithm"],
        "CV Accuracy": grid_search.best_score_,
        "Execution Time (s)": grid_time,
    },
    "RandomizedSearchCV": {
        "Best k": random_search.best_params_["n_neighbors"],
        "Metric": random_search.best_params_["metric"],
        "Weights": random_search.best_params_["weights"],
        "Algorithm": random_search.best_params_["algorithm"],
        "CV Accuracy": random_search.best_score_,
        "Execution Time (s)": random_time,
    },
})
search_comparison.to_csv(f"{RES_DIR}/gridsearch_vs_randomsearch.csv")
print("\n=== GridSearchCV vs RandomizedSearchCV ===")
print(search_comparison)

best_knn_params = grid_search.best_params_
best_knn = KNeighborsClassifier(**best_knn_params)
best_knn.fit(X_train_scaled, y_train)
y_pred_best_knn = best_knn.predict(X_test_scaled)
y_proba_best_knn = best_knn.predict_proba(X_test_scaled)

best_knn_metrics = classification_performance_metrics(
    y_test, y_pred_best_knn, y_proba_best_knn, model_name="Best KNN (Tuned)",
    return_dict=True, plot=True, save_path=f"{FIG_DIR}/cm_best_knn.eps"
)
print("\n=== Best (Tuned) KNN Metrics ===")
print(best_knn_metrics)

# GridSearchCV heatmap (k vs metric, mean of weights/algorithm)
cv_results = pd.DataFrame(grid_search.cv_results_)
pivot = cv_results.pivot_table(values="mean_test_score",
                                index="param_n_neighbors",
                                columns="param_metric", aggfunc="mean")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax,
            annot_kws={"fontfamily": "Times New Roman", "fontsize": 10})
_bold_axis_labels(ax, "Distance Metric", "k (n_neighbors)", "GridSearchCV Heatmap")
_save_eps(fig, f"{FIG_DIR}/gridsearch_heatmap.eps")
plt.close(fig)

# RandomizedSearchCV score distribution
random_cv_results = pd.DataFrame(random_search.cv_results_)
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(random_cv_results["mean_test_score"], kde=True, ax=ax, color="#8e44ad")
_bold_axis_labels(ax, "Mean CV Accuracy", "Frequency",
                   "RandomizedSearchCV Score Distribution")
_save_eps(fig, f"{FIG_DIR}/randomsearch_score_distribution.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 8. KDTREE vs BALLTREE
# --------------------------------------------------------------------------
tree_comparison = {}
for algo in ["kd_tree", "ball_tree"]:
    knn_tree = KNeighborsClassifier(n_neighbors=best_knn_params["n_neighbors"],
                                     weights=best_knn_params["weights"],
                                     metric=best_knn_params["metric"],
                                     algorithm=algo)
    y_pred, train_t, pred_t = time_fit_predict(knn_tree, X_train_scaled, X_test_scaled, y_train)
    tree_comparison[algo] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Training Time (s)": train_t,
        "Prediction Time (s)": pred_t,
    }
tree_comparison_df = pd.DataFrame(tree_comparison)
tree_comparison_df.to_csv(f"{RES_DIR}/kdtree_vs_balltree.csv")
print("\n=== KDTree vs BallTree ===")
print(tree_comparison_df)

# --------------------------------------------------------------------------
# 9. 5-FOLD CROSS VALIDATION (Naive Bayes best vs Best KNN)
# --------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

X_scaled_full = scaler.fit_transform(X)  # refit scaler on full X for CV convenience
nb_cv_scores = cross_val_score(GaussianNB() if best_nb_name == "Gaussian NB" else
                                (MultinomialNB() if best_nb_name == "Multinomial NB" else BernoulliNB()),
                                X if best_nb_name != "Gaussian NB" else X_scaled_full,
                                y, cv=skf, scoring="accuracy")
knn_cv_scores = cross_val_score(KNeighborsClassifier(**best_knn_params),
                                 X_scaled_full, y, cv=skf, scoring="accuracy")

cv_table = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(5)] + ["Average"],
    f"Naive Bayes ({best_nb_name})": list(nb_cv_scores) + [nb_cv_scores.mean()],
    "Best KNN": list(knn_cv_scores) + [knn_cv_scores.mean()],
}).set_index("Fold")
cv_table.to_csv(f"{RES_DIR}/cross_validation.csv")
print("\n=== 5-Fold Cross Validation ===")
print(cv_table)

fig, ax = plt.subplots(figsize=(7, 5))
folds = np.arange(1, 6)
ax.plot(folds, nb_cv_scores, marker="o", label=f"Naive Bayes ({best_nb_name})", color="#c0392b")
ax.plot(folds, knn_cv_scores, marker="s", label="Best KNN", color="#2980b9")
ax.legend()
_bold_axis_labels(ax, "Fold", "Accuracy", "Cross-Validation Accuracy")
_save_eps(fig, f"{FIG_DIR}/cv_accuracy.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 10. THEORETICAL vs EXPERIMENTAL TIME COMPLEXITY
# --------------------------------------------------------------------------
theoretical_complexity = pd.DataFrame({
    "Algorithm": ["Naive Bayes", "KNN (Brute)", "KDTree", "BallTree"],
    "Training": ["O(nd)", "O(1)", "O(n log n)", "O(n log n)"],
    "Prediction": ["O(d)", "O(nd)", "O(log n) avg", "O(log n) avg"],
}).set_index("Algorithm")
theoretical_complexity.to_csv(f"{RES_DIR}/theoretical_complexity.csv")

experimental_time = pd.DataFrame({
    "Gaussian NB": {"Training(s)": nb_timings["Gaussian NB"][0],
                    "Prediction(s)": nb_timings["Gaussian NB"][1]},
    "Multinomial NB": {"Training(s)": nb_timings["Multinomial NB"][0],
                       "Prediction(s)": nb_timings["Multinomial NB"][1]},
    "Bernoulli NB": {"Training(s)": nb_timings["Bernoulli NB"][0],
                     "Prediction(s)": nb_timings["Bernoulli NB"][1]},
}).T
experimental_time.to_csv(f"{RES_DIR}/experimental_time.csv")
print("\n=== Experimental Time Analysis (Naive Bayes) ===")
print(experimental_time)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
experimental_time["Training(s)"].plot(kind="bar", ax=axes[0], color="#16a085")
_bold_axis_labels(axes[0], "Model", "Time (s)", "Training Time Comparison")
experimental_time["Prediction(s)"].plot(kind="bar", ax=axes[1], color="#e67e22")
_bold_axis_labels(axes[1], "Model", "Time (s)", "Prediction Time Comparison")
plt.tight_layout()
_save_eps(fig, f"{FIG_DIR}/training_prediction_time.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 11. ROC CURVES, PRECISION-RECALL CURVES, CLASSIFIER COMPARISON
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for name, model, Xte in [
    ("Gaussian NB", nb_models["Gaussian NB"], X_test_scaled),
    ("Multinomial NB", nb_models["Multinomial NB"], X_test_raw),
    ("Bernoulli NB", nb_models["Bernoulli NB"], X_test_raw),
    ("Best KNN", best_knn, X_test_scaled),
]:
    proba = model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.8)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
ax.legend()
_bold_axis_labels(ax, "False Positive Rate", "True Positive Rate", "ROC Curves")
_save_eps(fig, f"{FIG_DIR}/roc_curves.eps")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
for name, model, Xte in [
    ("Gaussian NB", nb_models["Gaussian NB"], X_test_scaled),
    ("Multinomial NB", nb_models["Multinomial NB"], X_test_raw),
    ("Bernoulli NB", nb_models["Bernoulli NB"], X_test_raw),
    ("Best KNN", best_knn, X_test_scaled),
]:
    proba = model.predict_proba(Xte)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax.plot(rec, prec, label=name, linewidth=1.8)
ax.legend()
_bold_axis_labels(ax, "Recall", "Precision", "Precision-Recall Curves")
_save_eps(fig, f"{FIG_DIR}/precision_recall_curves.eps")
plt.close(fig)

all_models_summary = pd.concat([
    nb_results_df[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]],
    pd.DataFrame([best_knn_metrics]).set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]],
])
all_models_summary.to_csv(f"{RES_DIR}/classifier_comparison.csv")
print("\n=== Overall Classifier Comparison ===")
print(all_models_summary)

fig, ax = plt.subplots(figsize=(9, 5))
all_models_summary["Accuracy"].plot(kind="bar", ax=ax, color="#2c3e50")
_bold_axis_labels(ax, "Model", "Accuracy", "Classifier Comparison (Accuracy)")
plt.xticks(rotation=30, ha="right")
_save_eps(fig, f"{FIG_DIR}/classifier_comparison_bar.eps")
plt.close(fig)

# --------------------------------------------------------------------------
# 12. ADDITIONAL TASKS
# --------------------------------------------------------------------------
# (a) Different train-test splits
split_results = []
for test_size in [0.1, 0.2, 0.3, 0.4]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                           random_state=RANDOM_STATE, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    knn_tmp = KNeighborsClassifier(**best_knn_params)
    knn_tmp.fit(Xtr_s, ytr)
    acc = accuracy_score(yte, knn_tmp.predict(Xte_s))
    split_results.append({"Test Size": test_size, "Accuracy": acc})
split_df = pd.DataFrame(split_results).set_index("Test Size")
split_df.to_csv(f"{RES_DIR}/train_test_split_comparison.csv")
print("\n=== Train-Test Split Comparison (Best KNN) ===")
print(split_df)

# (b) Euclidean vs Manhattan distance
dist_results = {}
for metric in ["euclidean", "manhattan"]:
    knn_tmp = KNeighborsClassifier(n_neighbors=best_knn_params["n_neighbors"], metric=metric)
    knn_tmp.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn_tmp.predict(X_test_scaled))
    dist_results[metric] = acc
dist_df = pd.DataFrame([dist_results])
dist_df.to_csv(f"{RES_DIR}/distance_metric_comparison.csv")
print("\n=== Euclidean vs Manhattan Distance ===")
print(dist_df)

# (c) Weighted KNN (uniform vs distance)
weight_results = {}
for w in ["uniform", "distance"]:
    knn_tmp = KNeighborsClassifier(n_neighbors=best_knn_params["n_neighbors"], weights=w)
    knn_tmp.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn_tmp.predict(X_test_scaled))
    weight_results[w] = acc
weight_df = pd.DataFrame([weight_results])
weight_df.to_csv(f"{RES_DIR}/weighted_knn_comparison.csv")
print("\n=== Weighted KNN Comparison ===")
print(weight_df)

print("\nAll figures saved under:", os.path.abspath(FIG_DIR))
print("All result tables saved under:", os.path.abspath(RES_DIR))
print("\nDone.")
