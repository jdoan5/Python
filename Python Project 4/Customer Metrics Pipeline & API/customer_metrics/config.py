from pathlib import Path

# Directory: customer_metrics/config.py
# Resolve base directory of the package
BASE_DIR = Path(__file__).resolve().parent

# Where we'll store synthetic data and model artifacts
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Artifacts used by the API
MODEL_PATH = ARTIFACTS_DIR / "customer_churn_model.joblib"
FEATURE_COLUMNS_JSON = ARTIFACTS_DIR / "feature_columns.json"

# Columns must match CustomerFeatures in api/main.py
FEATURE_COLUMNS = [
    "age",
    "tenure_months",
    "avg_monthly_spend",
    "num_support_tickets",
    "is_premium",
]