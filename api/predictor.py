"""
api/predictor.py
~~~~~~~~~~~~~~~~
Loads saved model artefacts once at startup and exposes a clean
``predict`` / ``predict_batch`` interface used by the FastAPI routes.
"""

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from config.settings import (
    FEATURE_NAMES_FILENAME,
    MODEL_DIR,
    MODEL_FILENAME,
    SCALER_FILENAME,
    TOP_CATS_FILENAME,
)
from churn_predictor.features.engineering import transform_single_order

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Wrapper for the saved FLAML AutoML artefacts.
    Artefacts loaded
    ----------------
    - ``best_automl.pkl``      — trained AutoML model
    - ``feature_names.pkl``    — ordered list of column names
    - ``scaler.pkl``           — fitted StandardScaler
    - ``top_cats.pkl``         — list of kept product categories
    """

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        model_dir = Path(model_dir)
        logger.info("Loading model artefacts from %s", model_dir)

        self.automl        = joblib.load(model_dir / MODEL_FILENAME)
        self.feature_names = joblib.load(model_dir / FEATURE_NAMES_FILENAME)
        self.scaler        = joblib.load(model_dir / SCALER_FILENAME)
        self.top_cats      = joblib.load(model_dir / TOP_CATS_FILENAME)

        # Try to surface the underlying estimator name for /model/info
        try:
            self.estimator_name = self.automl.best_estimator
        except AttributeError:
            self.estimator_name = "unknown"
        self.variant_name = getattr(self.automl, "_settings", {}).get(
            "log_file_name", "unknown"
        )

        logger.info(
            "Model loaded — estimator: %s  features: %d",
            self.estimator_name,
            len(self.feature_names),
        )

    def predict(self, raw: dict) -> dict:
        """
        Predict churn for a single order.

        Parameters
        ----------
        raw : dict matching :class:`api.schemas.OrderFeatures`

        Returns
        -------
        dict with keys: churn_probability, is_churned, model_variant
        """
        X = transform_single_order(raw, self.scaler, self.feature_names, self.top_cats)
        proba = float(self.automl.predict_proba(X)[0, 1])
        label = int(self.automl.predict(X)[0])
        return {
            "churn_probability": round(proba, 6),
            "is_churned":        label,
            "model_variant":     self.estimator_name,
        }

    def predict_batch(self, raws: list[dict]) -> list[dict]:
        """
        Predict churn for a list of orders.
        Returns
        -------
        list of dicts matching the single-predict schema.
        """
        rows = [
            transform_single_order(r, self.scaler, self.feature_names, self.top_cats)
            for r in raws
        ]

        import pandas as pd
        X_batch = pd.concat(rows, ignore_index=True)

        probas = self.automl.predict_proba(X_batch)[:, 1]
        labels = self.automl.predict(X_batch)

        return [
            {
                "churn_probability": round(float(p), 6),
                "is_churned":        int(l),
                "model_variant":     self.estimator_name,
            }
            for p, l in zip(probas, labels)
        ]


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    """
    Return the singleton ChurnPredictor instance (loaded once per process).
    Called by FastAPI dependency injection.
    """
    return ChurnPredictor()
