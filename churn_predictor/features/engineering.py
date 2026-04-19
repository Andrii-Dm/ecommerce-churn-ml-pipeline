"""
churn_predictor/features/engineering.py

Two jobs:
  1. build_customer_features — full fit pipeline used during training,
     returns X, y, feature names, the fitted scaler, and top product categories
  2. transform_single_order — applies the saved artifacts to a single API
     request so inference uses exactly the same transformations as training
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.settings import (
    CATEGORICAL_FEATURES,
    DELIVERY_DELAY_CLIP,
    DELIVERY_TIME_CLIP,
    NUMERIC_FEATURES,
    TARGET_COL,
    TOP_N_CATEGORIES,
)

logger = logging.getLogger(__name__)


# -- geography tiers ---------------------------------------------------------
# city/state are too granular to use directly — bucket them by customer
# volume instead. qcut with q=3 can fail if there aren't enough distinct
# values (happens with states), so fall back to q=2 in that case

def _assign_tiers(counts_df, col, prefix):
    df = counts_df.copy()
    try:
        df[f"{prefix}_tier"] = pd.qcut(
            df[col], q=3,
            labels=[f"{prefix}_small", f"{prefix}_medium", f"{prefix}_large"],
        )
    except ValueError:
        df[f"{prefix}_tier"] = pd.qcut(
            df[col], q=2,
            labels=[f"{prefix}_small", f"{prefix}_large"],
        )
    return df


def _add_city_state_tiers(cust_level):
    city_counts = (
        cust_level.groupby("customer_city")["customer_unique_id"]
        .nunique().rename("city_customer_count").to_frame()
    )
    state_counts = (
        cust_level.groupby("customer_state")["customer_unique_id"]
        .nunique().rename("state_customer_count").to_frame()
    )

    city_counts  = _assign_tiers(city_counts,  "city_customer_count",  "city")
    state_counts = _assign_tiers(state_counts, "state_customer_count", "state")

    cust_level = cust_level.merge(
        city_counts[["city_tier"]], left_on="customer_city", right_index=True, how="left"
    )
    cust_level = cust_level.merge(
        state_counts[["state_tier"]], left_on="customer_state", right_index=True, how="left"
    )

    # categorical dtype from qcut needs the unknown label added before fillna
    for col, unknown in [("city_tier", "city_unknown"), ("state_tier", "state_unknown")]:
        if isinstance(cust_level[col].dtype, pd.CategoricalDtype):
            cust_level[col] = cust_level[col].cat.add_categories([unknown])
        cust_level[col] = cust_level[col].fillna(unknown)

    return cust_level


# -- main training pipeline --------------------------------------------------

def build_customer_features(orders_full, top_n_categories=TOP_N_CATEGORIES):
    """
    Builds the customer-level feature matrix from the merged Olist DataFrame.

    Only the first order per customer is used — the whole thesis framing is
    predicting repeat behavior from first-purchase signals, so anything
    beyond the first order would be data leakage.

    Returns X, y, feature_names, fitted scaler, top_cats
    """
    # one row per customer — take the earliest order by timestamp
    first_idx  = orders_full.groupby("customer_unique_id")["order_purchase_timestamp"].idxmin()
    cust_level = orders_full.loc[first_idx].copy()
    logger.info("Customer-level dataset: %d rows", len(cust_level))

    cust_level = _add_city_state_tiers(cust_level)

    # temporal features — weekday/hour as proxy for purchase intent type
    # (impulse buy at midnight vs planned purchase on a Tuesday afternoon)
    cust_level["purchase_weekday"] = cust_level["order_purchase_timestamp"].dt.weekday
    cust_level["purchase_hour"]    = cust_level["order_purchase_timestamp"].dt.hour

    # delivery features — two separate signals:
    # delivery_time_days = actual speed, delivery_delay_days = reliability vs promise
    cust_level["delivery_time_days"] = (
        cust_level["order_delivered_customer_date"] - cust_level["order_purchase_timestamp"]
    ).dt.days

    cust_level["delivery_delay_days"] = (
        cust_level["order_delivered_customer_date"] - cust_level["order_estimated_delivery_date"]
    ).dt.days

    for col in ["delivery_time_days", "delivery_delay_days"]:
        cust_level[col] = cust_level[col].fillna(cust_level[col].median())

    cust_level["delivery_time_days"]  = cust_level["delivery_time_days"].clip(*DELIVERY_TIME_CLIP)
    cust_level["delivery_delay_days"] = cust_level["delivery_delay_days"].clip(*DELIVERY_DELAY_CLIP)

    # keep top N product categories, collapse the long tail into "other"
    # this avoids explosion of dummy columns from hundreds of categories
    cust_level["main_product_cat_en"] = cust_level["main_product_cat_en"].fillna("unknown")
    top_cats = cust_level["main_product_cat_en"].value_counts().head(top_n_categories).index.tolist()
    cust_level["main_product_cat_en_top"] = np.where(
        cust_level["main_product_cat_en"].isin(top_cats),
        cust_level["main_product_cat_en"],
        "other",
    )

    cust_level["main_payment_type"] = cust_level["main_payment_type"].fillna("unknown")

    for col in NUMERIC_FEATURES:
        cust_level[col] = cust_level[col].fillna(cust_level[col].median())

    cat_dummies = pd.get_dummies(cust_level[CATEGORICAL_FEATURES], drop_first=True)

    scaler = StandardScaler()
    X_num = pd.DataFrame(
        scaler.fit_transform(cust_level[NUMERIC_FEATURES]),
        columns=NUMERIC_FEATURES,
        index=cust_level.index,
    )

    X = pd.concat([X_num, cat_dummies], axis=1).apply(pd.to_numeric, errors="raise").astype(float)
    # plain str column names — pandas 2.x get_dummies returns StringDtype
    # which breaks numpy's issubdtype inside FLAML
    X.columns = [str(c) for c in X.columns]
    feature_names = X.columns.tolist()

    y = cust_level[TARGET_COL].astype(int).values
    logger.info("Feature matrix shape: %s  |  churn rate: %.2f%%", X.shape, y.mean() * 100)

    return X, y, feature_names, scaler, top_cats


# -- inference transformer ---------------------------------------------------

def transform_single_order(raw, scaler, feature_names, top_cats):
    """
    Applies the same preprocessing to a single order dict for API inference.
    Needs to exactly mirror build_customer_features — any drift here will
    produce silently wrong predictions.
    """
    record = dict(raw)

    record["delivery_time_days"]  = float(np.clip(record.get("delivery_time_days",  0), *DELIVERY_TIME_CLIP))
    record["delivery_delay_days"] = float(np.clip(record.get("delivery_delay_days", 0), *DELIVERY_DELAY_CLIP))

    if record.get("main_product_cat_en_top") not in top_cats:
        record["main_product_cat_en_top"] = "other"

    numeric_vals = np.array([[record[col] for col in NUMERIC_FEATURES]], dtype=float)
    num_df = pd.DataFrame(scaler.transform(numeric_vals), columns=NUMERIC_FEATURES)

    cat_df = pd.get_dummies(
        pd.DataFrame({col: [record.get(col, "unknown")] for col in CATEGORICAL_FEATURES}),
        drop_first=True,
    )

    row = pd.concat([num_df, cat_df], axis=1)
    row.columns = [str(c) for c in row.columns]

    # align to training columns — missing dummies become 0, extras get dropped
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0.0

    return row[feature_names].astype(float)
