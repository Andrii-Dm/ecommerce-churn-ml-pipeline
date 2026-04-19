"""
churn_predictor/analysis/eda.py

EDA, econometric baseline (logit), and post-model interpretation.
All functions just write to disk — nothing returned, nothing fancy.

The logit here isn't the main model, it's the econometric sanity check
from chapter 2 — used to confirm direction/significance before throwing
AutoML at the problem.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from flaml import AutoML
from sklearn.inspection import permutation_importance

from config.settings import NUMERIC_FEATURES, PLOT_DIR, TABLE_DIR, TARGET_COL

logger = logging.getLogger(__name__)


# -- descriptive stats -------------------------------------------------------

def save_descriptive_stats(cust_level: pd.DataFrame) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)

    cust_level[NUMERIC_FEATURES + [TARGET_COL]].describe().to_csv(TABLE_DIR / "overall_summary.csv")
    logger.info("Saved overall_summary.csv")

    cust_level.isna().sum().to_csv(TABLE_DIR / "na_counts.csv", header=["na_count"])

    # compare one-time vs repeat buyers across all numeric features
    group_stats = (
        cust_level.groupby(TARGET_COL)[NUMERIC_FEATURES]
        .agg(["mean", "median", "std"])
        .T
    )
    group_stats.columns = ["Repeat (0)", "One-time (1)"]
    group_stats.round(4).to_csv(TABLE_DIR / "group_stats.csv")
    logger.info("Saved group_stats.csv")


# -- plots -------------------------------------------------------------------

def plot_churn_distribution(cust_level: pd.DataFrame) -> None:
    # 96%+ of customers are one-time buyers on Olist — this plot makes
    # the imbalance visually obvious, useful for the thesis figures
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.countplot(x=TARGET_COL, data=cust_level)
    plt.title("Customer distribution (1 = one-time, 0 = repeat)")
    plt.xlabel("is_churned")
    plt.ylabel("Count")
    plt.savefig(PLOT_DIR / "churn_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved churn_distribution.png")


def plot_correlation_matrix(cust_level: pd.DataFrame) -> None:
    # as expected, no single feature dominates — repeat purchase is
    # multivariate. total_price / avg_price / payment_value_sum cluster
    # together (they're economically the same thing essentially)
    os.makedirs(PLOT_DIR, exist_ok=True)
    corr = cust_level[NUMERIC_FEATURES + [TARGET_COL]].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation matrix — numeric features vs churn")
    plt.savefig(PLOT_DIR / "correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    corr.to_csv(TABLE_DIR / "correlation_matrix.csv")
    logger.info("Saved correlation_matrix.png")


def plot_violin_delivery(cust_level: pd.DataFrame) -> None:
    # delivery delay and time are among the stronger behavioral signals —
    # repeat buyers consistently see shorter times and fewer late deliveries
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    sns.violinplot(data=cust_level, x="is_churned", y="delivery_delay_days",
                   inner="quartile", ax=ax1)
    ax1.set_title("delivery_delay_days by churn group")

    sns.violinplot(data=cust_level, x="is_churned", y="delivery_time_days",
                   inner="quartile", ax=ax2)
    ax2.set_title("delivery_time_days by churn group")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "violin_delivery.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved violin_delivery.png")


# -- econometric baseline ----------------------------------------------------

def run_logistic_regression(X_numeric_scaled: pd.DataFrame, y: np.ndarray) -> None:
    """
    Logit on numeric features only — this is the chapter 2 econometric check,
    not the final model. Purpose is interpretability: which features are
    statistically significant and in which direction.

    McFadden R² came out ~0.01, confirming no single factor dominates
    and justifying the move to AutoML for the actual prediction task.
    """
    os.makedirs(TABLE_DIR, exist_ok=True)

    X_logit = sm.add_constant(
        pd.DataFrame(X_numeric_scaled.values, columns=NUMERIC_FEATURES),
        has_constant="add",
    ).astype(float)

    result = sm.Logit(y.astype(int), X_logit).fit(disp=False)

    pd.DataFrame({
        "coef":       result.params,
        "odds_ratio": np.exp(result.params),
        "p_value":    result.pvalues,
    }, index=["const"] + NUMERIC_FEATURES).round(6).to_csv(TABLE_DIR / "logit_coefficients.csv")

    logger.info("Logistic regression Pseudo R² (McFadden): %.4f", result.prsquared)
    logger.info("Saved logit_coefficients.csv")


# -- model interpretation ----------------------------------------------------

def plot_permutation_importance(best_automl, X_test, y_test, feature_names, variant_name, top_n=15):
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    logger.info("Computing permutation importance...")
    perm = permutation_importance(
        best_automl, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "feature":          feature_names,
        "importance_mean":  perm.importances_mean,
        "importance_std":   perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    perm_df.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)

    top = perm_df.head(top_n)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["importance_mean"][::-1])
    plt.xlabel("Accuracy drop on feature shuffle")
    plt.title(f"Permutation importance (top {top_n}) — {variant_name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "permutation_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved permutation_importance.png")


def plot_gain_importance(best_automl, feature_names, variant_name, top_n=15):
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    try:
        model = best_automl.model.estimator
        importances = model.feature_importances_

        # FLAML's internal transformer can drop/recode columns so the model's
        # feature count may differ from our original list — try to recover the
        # right names from the model itself before falling back to generic ones
        if hasattr(model, "feature_names_in_") and len(model.feature_names_in_) == len(importances):
            names = list(model.feature_names_in_)
        elif hasattr(model, "feature_name_") and len(model.feature_name_) == len(importances):
            names = list(model.feature_name_)  # lgbm stores it here
        elif len(feature_names) == len(importances):
            names = feature_names
        else:
            names = [f"f{i}" for i in range(len(importances))]

        gain_df = pd.DataFrame({
            "feature":         names,
            "gain_importance": importances,
        }).sort_values("gain_importance", ascending=False)

        gain_df.to_csv(TABLE_DIR / "gain_importance.csv", index=False)

        top = gain_df.head(top_n)
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"][::-1], top["gain_importance"][::-1])
        plt.xlabel("Gain importance")
        plt.title(f"Model gain importance (top {top_n}) — {variant_name}")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "gain_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved gain_importance.png")

    except AttributeError as e:
        logger.warning("Couldn't get feature_importances_ from model: %s", e)
