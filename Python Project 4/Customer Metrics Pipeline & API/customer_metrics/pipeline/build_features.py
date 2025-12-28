import pandas as pd
from customer_metrics.config import FEATURES_CSV
from customer_metrics.pipeline.ingest_customers import ingest_customers

FEATURE_COLUMNS = [
    "age",
    "tenure_months",
    "avg_monthly_spend",
    "num_support_tickets",
    "is_premium",
    # You can add engineered features here later, e.g.:
    # "spend_per_month",
]


def build_features() -> pd.DataFrame:
    df = ingest_customers()

    # Example engineered feature (commented out for now)
    # df["spend_per_month"] = df["avg_monthly_spend"] / (df["tenure_months"] + 1)

    features = df.copy()
    features.to_csv(FEATURES_CSV, index=False)
    print(f"Wrote features to {FEATURES_CSV}")

    return features


if __name__ == "__main__":
    build_features()