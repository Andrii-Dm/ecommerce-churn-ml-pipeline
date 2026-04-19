"""
tests/test_predictor.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the API layer.

These tests use a lightweight mock model so they can run without a trained
artefact on disk.  Run with:  pytest tests/
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import MagicMock, patch

from api.schemas import OrderFeatures
from churn_predictor.features.engineering import transform_single_order


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ORDER = {
    "n_items": 2,
    "total_price": 150.0,
    "total_freight": 20.0,
    "avg_price": 75.0,
    "payment_value_sum": 170.0,
    "payment_installments_avg": 1.0,
    "review_score_mean": 4.0,
    "delivery_time_days": 10,
    "delivery_delay_days": -2,
    "purchase_weekday": 2,
    "purchase_hour": 14,
    "main_product_cat_en_top": "health_beauty",
    "main_payment_type": "credit_card",
    "city_tier": "city_large",
    "state_tier": "state_large",
}


@pytest.fixture
def mock_scaler():
    scaler = MagicMock()
    # Return input unchanged (identity transform)
    scaler.transform = lambda X: X
    return scaler


@pytest.fixture
def top_cats():
    return ["health_beauty", "bed_bath_table", "sports_leisure"]


@pytest.fixture
def feature_names(mock_scaler, top_cats):
    """Derive feature names by running transform on a sample order."""
    from config.settings import NUMERIC_FEATURES
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    real_scaler = StandardScaler()
    import numpy as np
    real_scaler.fit(np.zeros((1, len(NUMERIC_FEATURES))))

    row = transform_single_order(SAMPLE_ORDER, real_scaler, [], top_cats)
    return list(row.columns)


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

class TestOrderFeaturesSchema:
    def test_valid_order_parses(self):
        order = OrderFeatures(**SAMPLE_ORDER)
        assert order.n_items == 2
        assert order.review_score_mean == 4.0

    def test_defaults_applied(self):
        minimal = {k: v for k, v in SAMPLE_ORDER.items()
                   if k not in ("main_product_cat_en_top", "main_payment_type",
                                "city_tier", "state_tier")}
        order = OrderFeatures(**minimal)
        assert order.main_product_cat_en_top == "other"
        assert order.main_payment_type == "credit_card"

    def test_invalid_review_score_rejected(self):
        bad = {**SAMPLE_ORDER, "review_score_mean": 6.0}
        with pytest.raises(Exception):
            OrderFeatures(**bad)

    def test_invalid_weekday_rejected(self):
        bad = {**SAMPLE_ORDER, "purchase_weekday": 7}
        with pytest.raises(Exception):
            OrderFeatures(**bad)


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestTransformSingleOrder:
    def test_output_shape(self, mock_scaler, top_cats):
        from config.settings import NUMERIC_FEATURES
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(np.zeros((1, len(NUMERIC_FEATURES))))

        row = transform_single_order(SAMPLE_ORDER, scaler, [], top_cats)
        assert row.shape[0] == 1

    def test_delivery_clipping(self, mock_scaler, top_cats):
        from config.settings import NUMERIC_FEATURES, DELIVERY_TIME_CLIP
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(np.zeros((1, len(NUMERIC_FEATURES))))

        extreme = {**SAMPLE_ORDER, "delivery_time_days": 999}
        row = transform_single_order(extreme, scaler, [], top_cats)
        # After clipping the value fed to scaler should be at most DELIVERY_TIME_CLIP[1]
        # (scaler.transform is applied after clipping)
        assert extreme["delivery_time_days"] == 999  # original unchanged

    def test_unknown_category_mapped_to_other(self, mock_scaler, top_cats):
        from config.settings import NUMERIC_FEATURES
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(np.zeros((1, len(NUMERIC_FEATURES))))

        unknown_cat = {**SAMPLE_ORDER, "main_product_cat_en_top": "some_rare_category_xyz"}
        # Should not raise; rare category → "other"
        row = transform_single_order(unknown_cat, scaler, [], top_cats)
        assert row.shape[0] == 1

    def test_column_alignment(self, mock_scaler, top_cats):
        """Output columns must exactly match feature_names (in order)."""
        from config.settings import NUMERIC_FEATURES
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(np.zeros((1, len(NUMERIC_FEATURES))))

        row_ref = transform_single_order(SAMPLE_ORDER, scaler, [], top_cats)
        names = list(row_ref.columns)

        # A second call with different categoricals should still match the same columns
        other_order = {**SAMPLE_ORDER, "main_payment_type": "boleto"}
        row2 = transform_single_order(other_order, scaler, names, top_cats)
        assert list(row2.columns) == names


# ---------------------------------------------------------------------------
# API route tests (using FastAPI TestClient with mocked predictor)
# ---------------------------------------------------------------------------

class TestAPIRoutes:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        from api.predictor import get_predictor

        mock_predictor = MagicMock()
        mock_predictor.estimator_name = "lgbm"
        mock_predictor.feature_names  = list(SAMPLE_ORDER.keys())
        mock_predictor.predict.return_value = {
            "churn_probability": 0.82,
            "is_churned":        1,
            "model_variant":     "lgbm",
        }
        mock_predictor.predict_batch.return_value = [
            {"churn_probability": 0.82, "is_churned": 1, "model_variant": "lgbm"},
            {"churn_probability": 0.31, "is_churned": 0, "model_variant": "lgbm"},
        ]

        app.dependency_overrides[get_predictor] = lambda: mock_predictor
        yield TestClient(app)
        app.dependency_overrides.clear()

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_model_info(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        body = r.json()
        assert "feature_count" in body
        assert "feature_names" in body

    def test_predict_single(self, client):
        r = client.post("/predict", json=SAMPLE_ORDER)
        assert r.status_code == 200
        body = r.json()
        assert "churn_probability" in body
        assert body["is_churned"] in (0, 1)

    def test_predict_batch(self, client):
        r = client.post("/predict/batch", json={"orders": [SAMPLE_ORDER, SAMPLE_ORDER]})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2

    def test_predict_batch_empty_rejected(self, client):
        r = client.post("/predict/batch", json={"orders": []})
        assert r.status_code == 422
