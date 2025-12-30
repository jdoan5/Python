import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MONTHLY_PATH = DATA_DIR / "transactions_monthly.csv"
RAW_PATH = DATA_DIR / "transactions_raw.csv"


# ---------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------
def generate_synthetic_data(
    n_customers: int = 60,
    start_month: str = "2022-01",
    end_month: str = "2023-12",
    random_seed: int = 42,
):
    rng = np.random.default_rng(random_seed)

    months = pd.period_range(start=start_month, end=end_month, freq="M")
    customers = [f"C{idx:03d}" for idx in range(1, n_customers + 1)]

    monthly_rows = []
    raw_rows = []

    categories = ["housing", "food", "transport", "shopping", "entertainment", "other"]

    for cust in customers:
        # baseline spending profile for this customer
        # (relative preferences per category)
        base_prefs = rng.dirichlet(np.ones(len(categories)) * 3.0)

        for month in months:
            # income between ~3000 and ~7000
            income = float(rng.normal(loc=4500, scale=800))
            income = max(income, 2500)  # keep it positive / reasonable

            # allocate a planned budget across categories
            planned_spend = income * 0.85  # assume 15% target savings
            base_spend = planned_spend * base_prefs

            # add noise to simulate under/over-spend per category
            noise = rng.normal(loc=1.0, scale=0.15, size=len(categories))
            actual_spend = np.maximum(base_spend * noise, 0.0)

            # compute totals and derived metrics
            total_spend = float(actual_spend.sum())
            savings = income - total_spend
            savings_rate = float(savings / income)
            overspend_flag = int(savings < 0)  # 1 if spending > income

            # monthly-level row (this is what the model will train on)
            row = {
                "customer_id": cust,
                "month": month.to_timestamp().strftime("%Y-%m"),
                "income": round(income, 2),
                "savings_rate": round(savings_rate, 3),
                "overspend_flag": overspend_flag,
            }
            for cat, amt in zip(categories, actual_spend):
                row[cat] = round(float(amt), 2)

            monthly_rows.append(row)

            # transaction-level rows (one per category; useful for future UX)
            tx_date = month.to_timestamp("M")  # last day of month
            for cat, amt in zip(categories, actual_spend):
                raw_rows.append(
                    {
                        "customer_id": cust,
                        "date": tx_date.strftime("%Y-%m-%d"),
                        "category": cat,
                        "amount": round(float(amt), 2),
                    }
                )

    monthly_df = pd.DataFrame(monthly_rows)
    raw_df = pd.DataFrame(raw_rows)

    return monthly_df, raw_df


def main():
    monthly_df, raw_df = generate_synthetic_data()

    monthly_df.to_csv(MONTHLY_PATH, index=False)
    raw_df.to_csv(RAW_PATH, index=False)

    print(f"Wrote {len(monthly_df)} rows to {MONTHLY_PATH}")
    print(f"Wrote {len(raw_df)} rows to {RAW_PATH}")


if __name__ == "__main__":
    main()