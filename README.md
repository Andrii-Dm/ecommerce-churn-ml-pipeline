# Customer Churn Prediction in E-Commerce
### Microservice-based ML pipeline for predicting repeat purchase behavior

> Master's thesis project — Yuriy Fedkovych Chernivtsi National University, 2025  
> Specialty: Economics / Information Technologies and Modelling in Economics  
> Author: Andrii Domin

> [!NOTE]
> The Research Summary below is an English summarization of the original thesis, which was written and submitted in Ukrainian. The full original text is not included in this repository.
>
> To improve accessibility and readability, I used AI-assisted tools to:
> - translate and refine documentation into English
> - structure and summarize the project for a broader audience
> - enhance inline comments and code clarity
>
> All technical design, implementation, and validation of results were done independently, with AI used only as a supporting tool for presentation and minor refactoring.
>
> The current repository reflects a refactored version of the original prototype, where I restructured the code into a more modular and maintainable form and introduced basic future-proofing.

---

## Research Summary

### The Problem

In e-commerce, acquiring a new customer costs significantly more than retaining an existing one. The ability to identify — at the moment of a customer's *first* purchase — whether they are likely to return is a high-value signal for marketing, logistics, and CRM teams. This project builds and deploys a machine learning model that does exactly that.

The research is grounded in the cybernetic view of CRM: the system must generate feedback signals about customer behavior and route them back into business decisions. Churn, in this model, is not just a loss event — it is a diagnostic indicator of where the system is failing.

### Dataset

The study uses the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — a real-world marketplace dataset covering 100,000+ orders across three years, with full transaction, logistics, payment, review, and customer geography data.

A key structural observation: **over 96% of Olist customers made only a single purchase**. This renders traditional time-gap churn definitions meaningless. Instead, churn is redefined as:

- `is_churned = 1` — customer made exactly one purchase (one-time buyer)
- `is_churned = 0` — customer made two or more purchases (repeat buyer)

This framing is both statistically appropriate for this dataset and practically meaningful: converting a first-time buyer into a repeat buyer is the central retention challenge in e-commerce.

### Feature Engineering

All features are derived exclusively from the **first order** — making the model usable in production immediately after a customer's first completed transaction. Four feature groups were constructed:

| Group | Features |
|---|---|
| Order structure | `n_items`, `total_price`, `avg_price`, `total_freight` |
| Payment behavior | `payment_value_sum`, `payment_installments_avg`, `main_payment_type` |
| Logistics & service | `delivery_time_days`, `delivery_delay_days`, `review_score_mean` |
| Behavioral timing | `purchase_weekday`, `purchase_hour` |
| Geography | `city_tier`, `state_tier`, `main_product_cat_en_top` |

### Econometric Baseline

Before machine learning, a logistic regression was fitted to assess the direction and significance of individual factors. Key findings:

- **More items in the basket** → higher probability of return
- **Lower total price and freight** → associated with repeat buyers
- **More installments used** → repeat buyers more commonly use financing options
- **Faster delivery, fewer delays** → strongly associated with return
- **Higher review score** → modest positive effect on repeat purchase
- Pseudo R² (McFadden): **0.0098** — confirming that no single factor dominates, and that the phenomenon is inherently multivariate

### Machine Learning Model

FLAML AutoML was used to search across CatBoost, XGBoost, LightGBM, and Random Forest. The core challenge is severe class imbalance (~97:3), so three balancing strategies were tested across multiple configurations:

| Variant | Accuracy | ROC AUC | F1 (repeat) | Combined Score |
|---|---|---|---|---|
| Baseline | 0.9700 | 0.5902 | 0.0000 | 0.5271 |
| Weighted 8:1 | 0.9663 | 0.6078 | 0.0248 | **0.5404** |
| Oversampled 0.15 | 0.9675 | 0.5439 | 0.0319 | 0.5174 |
| WeightSweep 12 | 0.9440 | 0.5916 | 0.0577 | 0.5371 |
| WeightSweep 20 | 0.8889 | 0.5881 | 0.0699 | 0.5229 |

The **best model is XGBoost with 8:1 class weighting**, selected by a combined score of `0.4 × ROC AUC + 0.3 × F1_repeat + 0.3 × Accuracy`. The baseline model identifies *zero* repeat buyers despite 97% accuracy — a classic imbalance failure. The weighted model sacrifices a small amount of accuracy in exchange for actually detecting the rare, high-value class.

### Model Interpretation (Permutation Importance)

The top predictive features, ranked by how much model accuracy drops when each feature is randomly shuffled:

1. `total_price` — the single strongest signal; cheaper first orders are more characteristic of repeat buyers
2. `payment_installments_avg` — use of installment financing is a strong marker of platform trust and engagement
3. `payment_value_sum` — higher total payment correlates with repeat behavior
4. `total_freight` — freight cost patterns differentiate the two groups
5. `delivery_time_days` — shorter delivery time is consistently associated with return

The model confirms the econometric findings but additionally captures nonlinear interactions between these features — something logistic regression cannot do.

### Key Business Conclusions

- **Repeat buyers are not random.** Their behavior differs systematically from the very first order.
- **Logistics is critical.** Even modest delivery delays measurably reduce the probability of return.
- **Installment usage signals intent.** Customers who invest effort in financing their first purchase are more likely to come back.
- **The model can operate from day one.** Since it uses only first-order features, predictions are available immediately after the first completed transaction — no waiting for behavioral history.
- **Integration is CRM-agnostic.** The microservice architecture means the model can be plugged into Odoo, Salesforce, Microsoft Dynamics, or any system capable of an HTTP POST request.

---

## Codebase

### Structure

```
churn_predictor/
├── config/
│   └── settings.py              # All paths, hyperparameters, and flags in one place
├── churn_predictor/
│   ├── data/loader.py           # Loads and merges the 7 Olist CSV tables
│   ├── features/engineering.py  # Feature engineering (training) + inference transform
│   ├── models/trainer.py        # FLAML experiment suite, model selection, artifact saving
│   └── analysis/eda.py          # EDA plots, logistic regression, permutation importance
├── api/
│   ├── main.py                  # FastAPI app
│   ├── predictor.py             # Model loading and inference logic
│   └── schemas.py               # Pydantic request/response models
├── scripts/
│   └── train.py                 # CLI entry-point for the full training pipeline
└── tests/
    └── test_predictor.py        # Schema, feature transform, and route tests
```

### Setup

```bash
pip install -r requirements.txt
```

Place the Olist CSVs in a `data/` directory at the project root. The expected files are:
```
data/
  olist_orders_dataset.csv
  olist_order_items_dataset.csv
  olist_customers_dataset.csv
  olist_order_payments_dataset.csv
  olist_order_reviews_dataset.csv
  olist_products_dataset.csv
  product_category_name_translation.csv
```

### Train

```bash
python scripts/train.py
# Skip EDA plots for faster runs:
python scripts/train.py --no-eda
```

Artifacts are saved to `model_artifacts/`: the trained model, scaler, feature names, and top product categories. Plots go to `plots/`, tables to `tables/`.

### Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "n_items": 2,
    "total_price": 89.90,
    "total_freight": 15.50,
    "avg_price": 44.95,
    "payment_value_sum": 105.40,
    "payment_installments_avg": 3,
    "review_score_mean": 5,
    "delivery_time_days": 8,
    "delivery_delay_days": -4,
    "purchase_weekday": 1,
    "purchase_hour": 11,
    "main_product_cat_en_top": "sports_leisure",
    "main_payment_type": "credit_card",
    "city_tier": "city_large",
    "state_tier": "state_large"
  }'
```

Response:
```json
{
  "churn_probability": 0.963,
  "is_churned": 1,
  "model_variant": "xgboost"
}
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Loaded model metadata and feature list |
| `POST` | `/predict` | Single-order churn prediction |
| `POST` | `/predict/batch` | Batch predictions (list of orders) |

### Configuration

Everything tunable lives in `config/settings.py`:

```python
FLAML_TIME_BUDGET = 300      # Seconds per experiment variant
TOP_N_CATEGORIES = 10        # Product categories kept as explicit features
WEIGHT_SWEEP_VALUES = [2, 4, 8, 12, 20, 30]   # Class weights to sweep over
```

### Dataset

The Olist dataset is not included in this repository. Download it from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the CSVs in the `data/` directory.
