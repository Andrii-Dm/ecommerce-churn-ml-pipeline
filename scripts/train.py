"""
scripts/train.py

Full training pipeline — loads Olist data, runs EDA, trains all FLAML
experiment variants, picks the best model and saves artifacts.

Usage:
    python scripts/train.py
    python scripts/train.py --no-eda          # skip plots, faster iteration
    python scripts/train.py --data-dir /path  # if data isn't in ./data/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- FLAML patch -------------------------------------------------------------
# pandas 2.x assigns StringDtype to column names produced by get_dummies,
# which numpy's issubdtype() can't interpret. Patch FLAML's DataTransformer
# to force object dtype on column index before it touches the data.
# Needs to happen before any other imports pull in flaml internals.
import flaml.automl.data as _flaml_data
_orig_fit_transform = _flaml_data.DataTransformer.fit_transform

def _patched_fit_transform(self, X, y, task, *args, **kwargs):
    if hasattr(X, "columns"):
        X = X.copy()
        X.columns = pd.Index([str(c) for c in X.columns], dtype=object)
    return _orig_fit_transform(self, X, y, task, *args, **kwargs)

_flaml_data.DataTransformer.fit_transform = _patched_fit_transform
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# logging has to be configured before project imports — flaml and sklearn
# both call basicConfig internally on import, so if we wait we lose control
# of the handlers and get no output
from config.settings import LOG_DIR
_log_dir = Path(LOG_DIR).resolve()
os.makedirs(_log_dir, exist_ok=True)

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_fmt  = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s - %(message)s")

_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_sh)

try:
    _fh = logging.FileHandler(_log_dir / "train.log", encoding="utf-8")
    _fh.setFormatter(_fmt)
    _root.addHandler(_fh)
except Exception as e:
    print(f"[warn] couldn't create log file: {e}", flush=True)

logger = logging.getLogger("train")

from config.settings import DATA_DIR, MODEL_DIR, PLOT_DIR, TABLE_DIR
from churn_predictor.data.loader import build_orders_full, load_raw_tables
from churn_predictor.features.engineering import build_customer_features
from churn_predictor.models.trainer import (
    plot_confusion_matrix,
    run_experiments,
    save_artefacts,
    select_best_model,
)
from churn_predictor.analysis.eda import (
    plot_churn_distribution,
    plot_correlation_matrix,
    plot_gain_importance,
    plot_permutation_importance,
    plot_violin_delivery,
    run_logistic_regression,
    save_descriptive_stats,
)


def main(data_dir, run_eda):
    for d in [MODEL_DIR, PLOT_DIR, TABLE_DIR]:
        os.makedirs(d, exist_ok=True)

    logger.info("=== Step 1: Loading data ===")
    orders_full = build_orders_full(load_raw_tables(data_dir))

    logger.info("=== Step 2: Feature engineering ===")
    X, y, feature_names, scaler, top_cats = build_customer_features(orders_full)

    # we need a test split here for the confusion matrix later — same seed
    # as run_experiments so the splits are identical
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if run_eda:
        logger.info("=== Step 3: EDA ===")
        # EDA needs cust_level (raw customer rows), not the scaled X matrix.
        # Rebuild it from orders_full rather than trying to invert the scaler.
        from churn_predictor.features.engineering import (
            _add_city_state_tiers, DELIVERY_DELAY_CLIP, DELIVERY_TIME_CLIP,
        )
        from config.settings import NUMERIC_FEATURES

        first_idx = orders_full.groupby("customer_unique_id")["order_purchase_timestamp"].idxmin()
        cl = orders_full.loc[first_idx].copy()
        cl = _add_city_state_tiers(cl)
        cl["purchase_weekday"]   = cl["order_purchase_timestamp"].dt.weekday
        cl["purchase_hour"]      = cl["order_purchase_timestamp"].dt.hour
        cl["delivery_time_days"] = (
            cl["order_delivered_customer_date"] - cl["order_purchase_timestamp"]
        ).dt.days.clip(*DELIVERY_TIME_CLIP).fillna(0)
        cl["delivery_delay_days"] = (
            cl["order_delivered_customer_date"] - cl["order_estimated_delivery_date"]
        ).dt.days.clip(*DELIVERY_DELAY_CLIP).fillna(0)

        save_descriptive_stats(cl)
        plot_churn_distribution(cl)
        plot_correlation_matrix(cl)
        plot_violin_delivery(cl)
        run_logistic_regression(X[NUMERIC_FEATURES], y)

    logger.info("=== Step 4: Training experiments ===")
    runs, results = run_experiments(X, y)

    logger.info("=== Step 5: Selecting best model ===")
    best_automl, best_row = select_best_model(runs, results)
    plot_confusion_matrix(best_automl, X_test, y_test, best_row["name"])

    logger.info("=== Step 6: Interpretation ===")
    plot_permutation_importance(best_automl, X_test, y_test, feature_names, best_row["name"])
    plot_gain_importance(best_automl, feature_names, best_row["name"])

    logger.info("=== Step 7: Saving artifacts ===")
    save_artefacts(best_automl, feature_names, scaler, top_cats)

    logger.info("Done. Best variant: %s", best_row["name"])
    logger.info(
        "acc=%.4f  auc=%.4f  f1_churn=%.4f  f1_repeat=%.4f  combined=%.4f",
        best_row["accuracy"], best_row["roc_auc"],
        best_row["f1_churn"], best_row["f1_repeat"], best_row["combined_score"],
    )


if __name__ == "__main__":
    print("train.py starting...", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-eda",   action="store_true")
    args = parser.parse_args()

    print(f"data_dir={args.data_dir.resolve()}  run_eda={not args.no_eda}", flush=True)

    try:
        main(data_dir=args.data_dir, run_eda=not args.no_eda)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)
