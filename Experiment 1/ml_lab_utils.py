"""
ml_lab_utils.py
================
ICS1512 - Machine Learning Algorithms Laboratory
Reusable utility module used across ALL experiments.

Implements (per lab manual, Section 4):
    1. One reusable EDA function            -> generate_eda_summary()
    2. One reusable Regression function      -> train_evaluate_regression()
    3. One reusable Classification function  -> train_evaluate_classification()
    4. One reusable Regression metrics fn    -> regression_performance_metrics()
    5. One reusable Classification metrics   -> classification_performance_metrics()

Formatting rules enforced everywhere (per lab manual, Section 1):
    - Times New Roman, 15 pt for all text / legends
    - Bold, Times New Roman, 15 pt axis labels
    - Figures exported as .eps at 600 DPI (Section 3)

NOTE on fonts: "Times New Roman" itself is a proprietary Microsoft font and is
not installable on Linux. Liberation Serif is metrically-compatible (identical
glyph widths/kerning) and is registered here under the family name
"Times New Roman" so that rcParams['font.family'] = 'Times New Roman' works
transparently. On Windows/macOS, if the real Times New Roman is installed,
matplotlib will simply use that instead.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 1. GLOBAL PLOT STYLE  (Section 1 of the manual)
# --------------------------------------------------------------------------
def set_plot_style(font_size=15):
    """
    Applies the mandatory lab formatting to every matplotlib figure:
        - Times New Roman (or metric-compatible Liberation Serif) font
        - 15 pt base font size
        - 15 pt Times New Roman legends
        - Bold, 15 pt, Times New Roman axis labels
    Call this once at the start of a notebook / script.
    """
    # Register Liberation Serif under the alias "Times New Roman" if the
    # genuine font is not present on this machine.
    installed_fonts = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" not in installed_fonts:
        liberation_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
        ]
        for p in liberation_paths:
            if os.path.exists(p):
                fm.fontManager.addfont(p)
                # Force the registered family name to "Times New Roman"
                # (FontEntry is a frozen dataclass in modern matplotlib, so we
                # replace the last-added entry rather than mutate it in place)
                last = fm.fontManager.ttflist[-1]
                fm.fontManager.ttflist[-1] = fm.FontEntry(
                    fname=last.fname, name="Times New Roman",
                    style=last.style, variant=last.variant,
                    weight=last.weight, stretch=last.stretch, size=last.size,
                )

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size,
        "axes.labelsize": font_size,
        "axes.labelweight": "bold",
        "axes.titlesize": font_size,
        "axes.titleweight": "bold",
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "figure.titlesize": font_size + 2,
        "savefig.dpi": 600,
        "figure.dpi": 150,   # screen preview; export always forced to 600 (see save)
        "svg.fonttype": "none",
    })


def _bold_axis_labels(ax, xlabel=None, ylabel=None, title=None, fs=15):
    """Helper: apply Times New Roman / Bold / 15pt to a single axis explicitly."""
    fp_bold = fm.FontProperties(family="Times New Roman", weight="bold", size=fs)
    fp_reg = fm.FontProperties(family="Times New Roman", size=fs - 2)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontproperties=fp_bold)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontproperties=fp_bold)
    if title is not None:
        ax.set_title(title, fontproperties=fp_bold, fontsize=fs)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(fp_reg)
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontproperties(fp_reg)


def _save_eps(fig, save_path):
    """Export a figure as .eps at 600 DPI (Section 3 of the manual)."""
    if save_path is None:
        return None
    if not save_path.lower().endswith(".eps"):
        save_path = os.path.splitext(save_path)[0] + ".eps"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, format="eps", dpi=600, bbox_inches="tight")
    return save_path


# --------------------------------------------------------------------------
# 2. GENERIC EDA FUNCTION  (Section 4.1)  -> ONE consolidated 12-subplot figure
# --------------------------------------------------------------------------
def generate_eda_summary(df, target_col=None, dataset_name="Dataset",
                          save_path=None, figsize=(22, 16)):
    """
    Generic, reusable EDA function that works on ANY tabular dataset
    (classification, regression, or unlabeled). Produces ONE consolidated
    figure containing 12 EDA subplots on a single page, per Section 2 of the
    lab manual.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset (features + target, if any).
    target_col : str or None
        Name of the target/label column, if present. If None, the function
        treats the dataset as unlabeled and adapts the 12-panel layout
        accordingly (no class-distribution / target-correlation panels).
    dataset_name : str
        Used in the figure's suptitle.
    save_path : str or None
        If given, the figure is exported as .eps @ 600 DPI to this path.
    figsize : tuple
        Overall figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    set_plot_style()
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    is_classification_target = (
        target_col is not None and
        (df[target_col].dtype == "object" or df[target_col].nunique() <= 20)
    )

    # Pick the most "informative" numeric feature (highest variance) as the
    # representative single feature for panels 6/8/9/10, instead of blindly
    # using the first column (which can be degenerate/constant, e.g. corner
    # pixels in an image dataset such as MNIST/Digits).
    if numeric_cols:
        # Prefer genuinely continuous columns (more than 5 distinct values) so
        # binary/near-constant encoded columns (e.g. a 0/1 "sex" flag, or
        # constant corner pixels in image data) are not picked as the
        # representative single feature for panels 6/8/9/10.
        continuous_cols = [c for c in numeric_cols if df[c].nunique() > 5]
        candidate_cols = continuous_cols if continuous_cols else numeric_cols
        variances = df[candidate_cols].var().sort_values(ascending=False)
        top_var_cols = variances.index.tolist()
        feat_a = top_var_cols[0]
        feat_b = top_var_cols[1] if len(top_var_cols) > 1 else top_var_cols[0]
        kde_cols = top_var_cols[:4]
    else:
        feat_a = feat_b = None
        kde_cols = []

    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"Exploratory Data Analysis Summary \u2013 {dataset_name}",
                 fontweight="bold", fontsize=17,
                 fontproperties=fm.FontProperties(family="Times New Roman",
                                                   weight="bold", size=17))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.4)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(12)]
    panel = 0

    # ---- Panel 1: Dataset overview (head / shape as a text table) ----
    ax = axes[panel]; panel += 1
    ax.axis("off")
    overview_txt = (
        f"Shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
        f"Numeric features: {len(numeric_cols)}\n"
        f"Categorical features: {len(categorical_cols)}\n"
        f"Missing cells: {int(df.isnull().sum().sum())}\n"
        f"Duplicate rows: {int(df.duplicated().sum())}"
    )
    ax.text(0.02, 0.9, overview_txt, va="top", ha="left",
            fontproperties=fm.FontProperties(family="Times New Roman", size=13),
            transform=ax.transAxes)
    _bold_axis_labels(ax, title="1. Dataset Overview")

    # ---- Panel 2: Statistical summary heat-table (mean/std/min/max) ----
    ax = axes[panel]; panel += 1
    if len(numeric_cols) > 0:
        desc = df[numeric_cols].describe().T[["mean", "std", "min", "max"]]
        desc_norm = (desc - desc.min()) / (desc.max() - desc.min() + 1e-9)
        sns.heatmap(desc_norm.iloc[:8], annot=desc.iloc[:8].round(1), fmt="",
                    cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"fontsize": 8, "fontfamily": "Times New Roman"})
    _bold_axis_labels(ax, title="2. Statistical Summary")

    # ---- Panel 3: Missing value analysis ----
    ax = axes[panel]; panel += 1
    miss = df.isnull().mean().sort_values(ascending=False) * 100
    if miss.sum() == 0:
        ax.text(0.5, 0.5, "No Missing Values", ha="center", va="center",
                fontproperties=fm.FontProperties(family="Times New Roman", size=14))
        ax.axis("off")
    else:
        miss[miss > 0].head(10).plot(kind="bar", ax=ax, color="#c0392b")
    _bold_axis_labels(ax, "Feature", "% Missing", "3. Missing Value Analysis")

    # ---- Panel 4: Class distribution / target distribution ----
    ax = axes[panel]; panel += 1
    if target_col is not None:
        if is_classification_target:
            df[target_col].value_counts().plot(kind="bar", ax=ax, color="#2980b9")
            _bold_axis_labels(ax, "Class", "Count", "4. Class Distribution")
        else:
            sns.histplot(df[target_col], kde=True, ax=ax, color="#2980b9")
            _bold_axis_labels(ax, target_col, "Frequency", "4. Target Distribution")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No target column supplied", ha="center", va="center",
                fontproperties=fm.FontProperties(family="Times New Roman", size=12))
        _bold_axis_labels(ax, title="4. Target Distribution")

    # ---- Panel 5: Correlation matrix (heatmap) ----
    ax = axes[panel]; panel += 1
    corr_cols = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
    if len(corr_cols) >= 2:
        sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", center=0, ax=ax,
                    cbar=False, annot=len(corr_cols) <= 6, fmt=".2f",
                    annot_kws={"fontsize": 7})
    _bold_axis_labels(ax, title="5. Correlation Matrix")

    # ---- Panel 6: Feature distribution (histogram of 1st numeric feature) ----
    ax = axes[panel]; panel += 1
    if feat_a is not None:
        sns.histplot(df[feat_a], kde=True, ax=ax, color="#27ae60")
    _bold_axis_labels(ax, feat_a if feat_a else "", "Frequency",
                       "6. Feature Distribution")

    # ---- Panel 7: Box plot (outlier detection) across numeric features ----
    ax = axes[panel]; panel += 1
    if len(numeric_cols) > 0:
        plot_cols = numeric_cols[:6]
        df_scaled = (df[plot_cols] - df[plot_cols].mean()) / (df[plot_cols].std() + 1e-9)
        sns.boxplot(data=df_scaled, ax=ax, color="#f39c12")
        ax.tick_params(axis="x", rotation=45)
    _bold_axis_labels(ax, "Feature", "Standardized Value", "7. Box Plot (Outliers)")

    # ---- Panel 8: Violin plot ----
    ax = axes[panel]; panel += 1
    if feat_a is not None:
        sns.violinplot(y=df[feat_a], ax=ax, color="#8e44ad")
    _bold_axis_labels(ax, "", feat_a if feat_a else "",
                       "8. Violin Plot")

    # ---- Panel 9: Scatter plot (feature 1 vs feature 2, hued by target) ----
    ax = axes[panel]; panel += 1
    if feat_a is not None and feat_b is not None:
        hue = df[target_col] if (target_col and is_classification_target) else None
        sns.scatterplot(x=df[feat_a], y=df[feat_b],
                         hue=hue, ax=ax, palette="Set2", legend=False, s=18)
    _bold_axis_labels(ax, feat_a if feat_a else "", feat_b if feat_b else "",
                       "9. Scatter Plot")

    # ---- Panel 10: Q-Q plot (normality check on highest-variance feature) ----
    ax = axes[panel]; panel += 1
    if feat_a is not None:
        stats.probplot(df[feat_a].dropna(), dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor("#2980b9")
        ax.get_lines()[0].set_markeredgecolor("#2980b9")
        ax.get_lines()[1].set_color("#c0392b")
    _bold_axis_labels(ax, "Theoretical Quantiles", "Sample Quantiles", "10. Q-Q Plot")

    # ---- Panel 11: KDE / density plot overlay of top numeric features ----
    ax = axes[panel]; panel += 1
    for c in kde_cols:
        sns.kdeplot(df[c], ax=ax, label=c, linewidth=1.5)
    if kde_cols:
        ax.legend(prop=fm.FontProperties(family="Times New Roman", size=9))
    _bold_axis_labels(ax, "Value", "Density", "11. KDE / Density Plot")

    # ---- Panel 12: Feature importance / variance plot ----
    ax = axes[panel]; panel += 1
    if len(numeric_cols) > 0:
        var = df[numeric_cols].var().sort_values(ascending=False).head(8)
        var.plot(kind="barh", ax=ax, color="#16a085")
        ax.invert_yaxis()
    _bold_axis_labels(ax, "Variance", "Feature", "12. Variance / Importance Plot")

    for ax in axes:
        _bold_axis_labels(ax)  # re-apply tick font in case a plotting call reset it

    saved = _save_eps(fig, save_path)
    if saved:
        print(f"[generate_eda_summary] Figure saved -> {saved} (600 DPI, EPS)")
    return fig


# --------------------------------------------------------------------------
# 3. GENERIC REGRESSION TRAIN/EVAL FUNCTION  (Section 4.2)
# --------------------------------------------------------------------------
def train_evaluate_regression(models: dict, X_train, X_test, y_train, y_test,
                               scale=False, verbose=True):
    """
    Reusable function to train and evaluate an arbitrary set of regression
    models on the same train/test split.

    Parameters
    ----------
    models : dict {name: sklearn-estimator}
    X_train, X_test, y_train, y_test : array-like
    scale : bool -> StandardScaler applied when True (fit on train only)
    verbose : bool -> print per-model metrics as they are computed

    Returns
    -------
    results_df : pandas.DataFrame  (one row per model, sorted by R2 desc)
    fitted_models : dict {name: fitted estimator}
    """
    from sklearn.preprocessing import StandardScaler

    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    rows, fitted_models = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = regression_performance_metrics(y_test, y_pred, model_name=name,
                                                   verbose=verbose, return_dict=True)
        rows.append(metrics)
        fitted_models[name] = model

    results_df = pd.DataFrame(rows).set_index("Model").sort_values("R2", ascending=False)
    return results_df, fitted_models


# --------------------------------------------------------------------------
# 4. GENERIC CLASSIFICATION TRAIN/EVAL FUNCTION  (Section 4.3)
# --------------------------------------------------------------------------
def train_evaluate_classification(models: dict, X_train, X_test, y_train, y_test,
                                   scale=False, average="weighted", verbose=True):
    """
    Reusable function to train and evaluate an arbitrary set of classification
    models on the same train/test split.

    Returns
    -------
    results_df : pandas.DataFrame  (one row per model, sorted by Accuracy desc)
    fitted_models : dict {name: fitted estimator}
    """
    from sklearn.preprocessing import StandardScaler

    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    rows, fitted_models = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                y_proba = None
        metrics = classification_performance_metrics(
            y_test, y_pred, y_proba=y_proba, model_name=name,
            average=average, verbose=verbose, return_dict=True, plot=False
        )
        rows.append(metrics)
        fitted_models[name] = model

    results_df = pd.DataFrame(rows).set_index("Model").sort_values("Accuracy", ascending=False)
    return results_df, fitted_models


# --------------------------------------------------------------------------
# 5. GENERIC REGRESSION METRICS FUNCTION  (Section 4.4)
# --------------------------------------------------------------------------
def regression_performance_metrics(y_true, y_pred, model_name="Model",
                                    verbose=True, return_dict=False):
    """
    Computes and displays ALL standard regression performance metrics:
    MAE, MSE, RMSE, R2, Adjusted R2 (n only), MAPE.
    """
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                  r2_score, mean_absolute_percentage_error)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    if verbose:
        print(f"--- Regression Metrics: {model_name} ---")
        print(f"  MAE  : {mae:.4f}")
        print(f"  MSE  : {mse:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")
        print(f"  MAPE : {mape:.2f}%\n")

    result = {"Model": model_name, "MAE": mae, "MSE": mse,
              "RMSE": rmse, "R2": r2, "MAPE(%)": mape}
    return result if return_dict else pd.DataFrame([result]).set_index("Model")


# --------------------------------------------------------------------------
# 6. GENERIC CLASSIFICATION METRICS FUNCTION  (Section 4.5)
# --------------------------------------------------------------------------
def classification_performance_metrics(y_true, y_pred, y_proba=None,
                                        model_name="Model", average="weighted",
                                        verbose=True, return_dict=False,
                                        plot=True, save_path=None):
    """
    Computes and displays ALL standard classification performance metrics:
    Accuracy, Precision, Recall, F1-score, ROC-AUC (binary/multiclass ovr),
    and (optionally) plots the confusion matrix using the mandatory lab
    formatting (Times New Roman, bold 15pt axis labels).
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, confusion_matrix)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    roc_auc = np.nan
    if y_proba is not None:
        try:
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                         average=average)
        except Exception:
            roc_auc = np.nan

    if verbose:
        print(f"--- Classification Metrics: {model_name} ---")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}" if not np.isnan(roc_auc) else "  ROC-AUC   : N/A")
        print()

    if plot:
        set_plot_style()
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"fontfamily": "Times New Roman", "fontsize": 13})
        _bold_axis_labels(ax, "Predicted Label", "True Label",
                           f"Confusion Matrix \u2013 {model_name}")
        _save_eps(fig, save_path)

    result = {"Model": model_name, "Accuracy": acc, "Precision": prec,
              "Recall": rec, "F1-score": f1, "ROC-AUC": roc_auc}
    return result if return_dict else pd.DataFrame([result]).set_index("Model")
