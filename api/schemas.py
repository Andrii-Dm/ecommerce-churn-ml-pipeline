"""
api/schemas.py
~~~~~~~~~~~~~~
Pydantic request and response models for the churn prediction API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class OrderFeatures(BaseModel):
    """
    Input features for a single order's churn prediction.

    All fields correspond to the features produced by the training pipeline.
    Categorical fields accept the string values used during training; unknown
    values are handled gracefully by the predictor.
    """

    # --- Numeric ---
    n_items: float                    = Field(..., ge=1,    description="Number of distinct items in the order")
    total_price: float                = Field(..., ge=0,    description="Sum of item prices (BRL)")
    total_freight: float              = Field(..., ge=0,    description="Sum of freight values (BRL)")
    avg_price: float                  = Field(..., ge=0,    description="Average item price (BRL)")
    payment_value_sum: float          = Field(..., ge=0,    description="Total payment amount (BRL)")
    payment_installments_avg: float   = Field(..., ge=1,    description="Average number of installments")
    review_score_mean: float          = Field(..., ge=1, le=5, description="Mean review score (1–5)")
    delivery_time_days: float         = Field(...,          description="Actual delivery time in days")
    delivery_delay_days: float        = Field(...,          description="Delivery delay vs estimated (negative = early)")
    purchase_weekday: int             = Field(..., ge=0, le=6, description="Day of week (0=Mon … 6=Sun)")
    purchase_hour: int                = Field(..., ge=0, le=23, description="Hour of day (0–23)")

    # --- Categorical ---
    main_product_cat_en_top: str      = Field("other",        description="Top product category (English)")
    main_payment_type: str            = Field("credit_card",  description="Dominant payment type")
    city_tier: str                    = Field("city_large",   description="Customer city size tier")
    state_tier: str                   = Field("state_large",  description="Customer state size tier")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response for a single-order churn prediction."""
    churn_probability: float  = Field(..., description="Probability of being a one-time buyer (0–1)")
    is_churned: int           = Field(..., description="Binary prediction: 1 = likely churner, 0 = likely repeat")
    model_variant: str        = Field(..., description="Name of the best-selected model variant")


class BatchPredictionRequest(BaseModel):
    orders: list[OrderFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class ModelInfo(BaseModel):
    variant: str
    feature_count: int
    feature_names: list[str]
    estimator: Optional[str] = None
