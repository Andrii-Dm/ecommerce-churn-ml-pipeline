"""
churn_predictor/models/trainer.py

FLAML experiment runner for the repeat-purchase churn task.
Handles class imbalance via weight sweep + oversampling,
picks the best model by a custom combined score, and dumps artifacts.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flaml import AutoML
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config.settings import (
    BASE_REPEAT_WEIGHT,
    COMBINED_SCORE_WEIGHTS,
    FEATURE_NAMES_FILENAME,
    FLAML_ESTIMATORS,
    FLAML_SWEEP_TIME_BUDGET,
    FLAML_TIME_BUDGET,
    LOG_DIR,
    MODEL_DIR,
    MODEL_FILENAME,
    OVERSAMPLE_RATIO,
    PLOT_DIR,
    SCALER_FILENAME,
    TABLE_DIR,
    TOP_CATS_FILENAME,
    WEIGHT_SWEEP_VALUES,
)

logger = logging.getLogger(__name__)


def _run_flaml(name, X_tr, y_tr, X_te, y_te, sw=None, budget=FLAML_TIME_BUDGET):
    """
    Fit one FLAML run and return metrics + the automl object.
    sw = per-sample weights array, or None for uniform weighting.
    """
    logger.info("Training variant: %s", name)
    os.makedirs(LOG_DIR, exist_ok=True)

    # pandas 2.x get_dummies produces StringDtype column names which
    # numpy can't interpret in issubdtype() — force plain str here
    X_tr = X_tr.copy()
    X_tr.columns = [str(c) for c in X_tr.columns]
    X_te = X_te.copy()
    X_te.columns = [str(c) for c in X_te.columns]

    automl = AutoML()
    cfg = {
        "time_budget":    budget,
        "metric":         "f1",
        "task":           "classification",
        "log_file_name":  str(LOG_DIR / f"flaml_{name}.log"),
        "verbose":        0,
        "estimator_list": FLAML_ESTIMATORS,
    }

    if sw is not None:
        automl.fit(X_tr, y_tr, sample_weight=sw, **cfg)
    else:
        automl.fit(X_tr, y_tr, **cfg)

    preds  = automl.predict(X_te)
    probas = automl.predict_proba(X_te)[:, 1]

    out = {
        "name":      name,
        "accuracy":  accuracy_score(y_te, preds),
        "roc_auc":   roc_auc_score(y_te, probas),
        # f1 for the churned class (majority) and repeat class (minority, the one we care about)
        "f1_churn":  f1_score(y_te, preds, pos_label=1),
        "f1_repeat": f1_score(y_te, preds, pos_label=0),
        "automl":    automl,
    }

    logger.info(
        "[%s] acc=%.4f  auc=%.4f  f1_churn=%.4f  f1_repeat=%.4f",
        name, out["accuracy"], out["roc_auc"], out["f1_churn"], out["f1_repeat"],
    )
    return out


def run_experiments(X: pd.DataFrame, y: np.ndarray):
    # Rebuild X from scratch so pandas StringDtype index metadata doesn't
    # survive FLAML's internal slicing (assigning .columns isn't enough)
    X = pd.DataFrame(X.values.astype(float), columns=[str(c) for c in X.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info("Train: %s  Test: %s", X_train.shape, X_test.shape)

    runs = []

    # --- baseline: no balancing, just to see where we start ---
    runs.append(_run_flaml("baseline", X_train, y_train, X_test, y_test))

    # --- single soft-weight run for a quick sanity check before the sweep ---
    sw = np.where(y_train == 0, BASE_REPEAT_WEIGHT, 1.0)
    runs.append(_run_flaml(
        f"weighted_{int(BASE_REPEAT_WEIGHT)}",
        X_train, y_train, X_test, y_test, sw=sw,
    ))

    # --- oversampling: tends to overfit but worth having for comparison ---
    ros = RandomOverSampler(sampling_strategy=OVERSAMPLE_RATIO, random_state=42)
    X_over, y_over = ros.fit_resample(X_train, y_train)
    runs.append(_run_flaml(
        f"oversampled_{OVERSAMPLE_RATIO}",
        X_over, y_over, X_test, y_test,
    ))

    # --- weight sweep: the main experiment, trying several repeat-class weights ---
    for w in WEIGHT_SWEEP_VALUES:
        sw = np.where(y_train == 0, w, 1.0)
        exp = _run_flaml(
            f"weightSweep_{w}",
            X_train, y_train, X_test, y_test,
            sw=sw, budget=FLAML_SWEEP_TIME_BUDGET,
        )
        exp["weight"] = w
        runs.append(exp)

    # drop automl objects before building the comparison table
    records = [{"exp_id": i, **{k: v for k, v in r.items() if k != "automl"}}
               for i, r in enumerate(runs)]

    results = pd.DataFrame(records)
    ws = COMBINED_SCORE_WEIGHTS
    results["combined_score"] = (
        ws["roc_auc"]   * results["roc_auc"]
        + ws["f1_repeat"] * results["f1_repeat"]
        + ws["accuracy"]  * results["accuracy"]
    )

    os.makedirs(TABLE_DIR, exist_ok=True)
    results.to_csv(TABLE_DIR / "flaml_model_comparison.csv", index=False)
    logger.info("Model comparison table saved.")

    return runs, results


def select_best_model(runs, results):
    best_idx    = int(results["combined_score"].idxmax())
    best_row    = results.loc[best_idx].to_dict()
    best_automl = runs[int(best_row["exp_id"])]["automl"]
    logger.info("Best variant: %s  combined_score=%.4f", best_row["name"], best_row["combined_score"])
    return best_automl, best_row


def save_artefacts(best_automl, feature_names, scaler, top_cats):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_automl,   MODEL_DIR / MODEL_FILENAME)
    joblib.dump(feature_names, MODEL_DIR / FEATURE_NAMES_FILENAME)
    joblib.dump(scaler,        MODEL_DIR / SCALER_FILENAME)
    joblib.dump(top_cats,      MODEL_DIR / TOP_CATS_FILENAME)
    logger.info("Artefacts saved to %s", MODEL_DIR)


def plot_confusion_matrix(best_automl, X_test, y_test, variant_name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, best_automl.predict(X_test))
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {variant_name}")
    plt.savefig(PLOT_DIR / "confusion_matrix_best_model.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved.")
