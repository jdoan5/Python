import pandas as pd
from customer_metrics.config import RAW_CUSTOMERS_CSV

REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "tenure_months",
    "avg_monthly_spend",
    "num_support_tickets",
    "is_premium",
    "churned",
]


def ingest_customers() -> pd.DataFrame:
    if not RAW_CUSTOMERS_CSV.exists():
        raise FileNotFoundError(f"Raw customers file not found: {RAW_CUSTOMERS_CSV}")

    df = pd.read_csv(RAW_CUSTOMERS_CSV)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Basic type cleaning
    df["age"] = df["age"].astype(int)
    df["tenure_months"] = df["tenure_months"].astype(int)
    df["avg_monthly_spend"] = df["avg_monthly_spend"].astype(float)
    df["num_support_tickets"] = df["num_support_tickets"].astype(int)
    df["is_premium"] = df["is_premium"].astype(int)
    df["churned"] = df["churned"].astype(int)

    # Drop rows with obvious issues
    df = df.dropna(subset=REQUIRED_COLUMNS)

    return df


if __name__ == "__main__":
    df = ingest_customers()
    print(df.head())
    print(f"Ingested {len(df)} rows from {RAW_CUSTOMERS_CSV}")