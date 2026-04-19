"""
config/settings.py
~~~~~~~~~~~~~~~~~~
Central configuration for the Churn Predictor pipeline.
All paths and hyper-parameters live here; nothing is hard-coded elsewhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR        = ROOT_DIR / "data"
PLOT_DIR        = ROOT_DIR / "plots"
TABLE_DIR       = ROOT_DIR / "tables"
MODEL_DIR       = ROOT_DIR / "model_artifacts"
LOG_DIR         = ROOT_DIR / "logs"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
TOP_N_CATEGORIES: int = 10          # keep top-N product categories, rest → "other"

DELIVERY_TIME_CLIP   = (0, 60)      # (min, max) days
DELIVERY_DELAY_CLIP  = (-20, 40)    # (min, max) days

NUMERIC_FEATURES = [
    "n_items",
    "total_price",
    "total_freight",
    "avg_price",
    "payment_value_sum",
    "payment_installments_avg",
    "review_score_mean",
    "delivery_time_days",
    "delivery_delay_days",
    "purchase_weekday",
    "purchase_hour",
]

CATEGORICAL_FEATURES = [
    "main_product_cat_en_top",
    "main_payment_type",
    "city_tier",
    "state_tier",
]

TARGET_COL = "is_churned"

# ---------------------------------------------------------------------------
# FLAML training
# ---------------------------------------------------------------------------
FLAML_TIME_BUDGET: int     = 300    # seconds per experiment variant
FLAML_SWEEP_TIME_BUDGET: int = 200  # shorter budget for weight-sweep runs
FLAML_ESTIMATORS           = ["catboost", "lgbm", "rf", "xgboost"]

WEIGHT_SWEEP_VALUES        = [2, 4, 8, 12, 20, 30]
BASE_REPEAT_WEIGHT: float  = 8.0
OVERSAMPLE_RATIO: float    = 0.15   # RandomOverSampler sampling_strategy

# ---------------------------------------------------------------------------
# Model selection combined score
# ---------------------------------------------------------------------------
COMBINED_SCORE_WEIGHTS = {
    "roc_auc":   0.4,
    "f1_repeat": 0.3,
    "accuracy":  0.3,
}

# ---------------------------------------------------------------------------
# Saved artefact filenames
# ---------------------------------------------------------------------------
MODEL_FILENAME        = "best_automl.pkl"
FEATURE_NAMES_FILENAME = "feature_names.pkl"
SCALER_FILENAME       = "scaler.pkl"
TOP_CATS_FILENAME     = "top_cats.pkl"
