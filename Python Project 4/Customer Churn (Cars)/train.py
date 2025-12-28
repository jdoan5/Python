import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib


# ---------------------------------------------------------
# 1. Synthetic data generator
# ---------------------------------------------------------

def generate_synthetic_churn_data(n=2000, random_state=42):
    """
    Create a synthetic customer churn dataset for car models.
    One text column 'text' will be used for TF-IDF.
    """
    rng = np.random.default_rng(random_state)

    customer_id = np.arange(1, n + 1)

    car_models = [
        "Sedan Basic",
        "Sedan Premium",
        "SUV Family",
        "SUV Sport",
        "Truck Utility",
        "EV Compact",
    ]
    contract_types = ["lease", "finance", "subscription"]

    car_model = rng.choice(car_models, size=n)
    contract_type = rng.choice(contract_types, size=n, p=[0.4, 0.4, 0.2])
    tenure_months = rng.integers(1, 60, size=n)  # how long they've had the car
    service_calls_last_3m = rng.poisson(lam=2, size=n)
    satisfaction_score = rng.integers(1, 6, size=n)  # 1–5
    monthly_payment = rng.normal(400, 120, size=n).clip(150, 1200)
    complaint_level = rng.integers(0, 4, size=n)  # 0 none, 3 heavy complaints

    # build a simple underlying churn probability (logistic model)
    tenure_scaled = (tenure_months - 24) / 12.0
    payment_scaled = (monthly_payment - 400) / 120.0
    is_subscription = (contract_type == "subscription").astype(int)

    logit = (
        -1.0
        - 0.3 * tenure_scaled
        + 0.4 * service_calls_last_3m
        - 0.7 * (satisfaction_score - 3)
        + 0.3 * payment_scaled
        + 0.8 * complaint_level
        + 0.5 * is_subscription
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    # build a text description for TF-IDF
    def build_text(i):
        pieces = []

        # car model
        pieces.append(f"Customer drives a {car_model[i]}.")

        # satisfaction-based phrase
        if satisfaction_score[i] <= 2:
            pieces.append("Customer is very unhappy with the car.")
        elif satisfaction_score[i] == 3:
            pieces.append("Customer is neutral about the car.")
        else:
            pieces.append("Customer is satisfied and generally likes the car.")

        # service complaints
        if service_calls_last_3m[i] == 0:
            pieces.append("No recent service issues or complaints.")
        elif service_calls_last_3m[i] < 3:
            pieces.append("Some minor service issues were reported recently.")
        else:
            pieces.append("Frequent service complaints about reliability and repairs.")

        # payment sentiment
        if monthly_payment[i] > 600:
            pieces.append("Customer says the monthly payment feels too expensive.")
        elif monthly_payment[i] < 300:
            pieces.append("Customer considers the monthly payment affordable.")

        # contract type phrases
        if contract_type[i] == "subscription":
            pieces.append("Customer is on a flexible subscription contract.")
        elif contract_type[i] == "lease":
            pieces.append("Customer is currently leasing the vehicle.")
        else:
            pieces.append("Customer is financing the car.")

        # complaint level specific text
        if complaint_level[i] == 3:
            pieces.append("Customer mentions switching to another brand soon.")
        elif complaint_level[i] == 2:
            pieces.append("Customer is considering alternatives due to recurring issues.")
        elif complaint_level[i] == 0:
            pieces.append("Customer expresses loyalty to the brand.")

        return " ".join(pieces)

    text = [build_text(i) for i in range(n)]

    df = pd.DataFrame({
        "customer_id": customer_id,
        "car_model": car_model,
        "contract_type": contract_type,
        "tenure_months": tenure_months,
        "service_calls_last_3m": service_calls_last_3m,
        "satisfaction_score": satisfaction_score,
        "monthly_payment": monthly_payment.round(2),
        "complaint_level": complaint_level,
        "text": text,
        "churn": churn,
    })

    # a tiny bit of messiness: randomly drop some text
    mask_missing_text = rng.random(n) < 0.02
    df.loc[mask_missing_text, "text"] = "Customer did not provide any comments."

    return df


# ---------------------------------------------------------
# 2. Cost-aware threshold search
# ---------------------------------------------------------

def find_best_threshold(y_true, y_proba, cost_fn=5.0, cost_fp=1.0):
    """
    Find probability threshold minimizing cost:
      cost = cost_fn * FN + cost_fp * FP
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr = 0.5
    best_cost = float("inf")

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost = cost
            best_thr = thr

    return best_thr, best_cost


# ---------------------------------------------------------
# 3. Training pipeline
# ---------------------------------------------------------

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1) generate synthetic data
    df = generate_synthetic_churn_data(n=3000)
    df.to_csv("data/customers_churn_synthetic.csv", index=False)
    print("Saved synthetic dataset to data/customers_churn_synthetic.csv")

    X = df["text"]
    y = df["churn"]

    # 2) train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # 3) TF-IDF + Logistic Regression baseline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000
        )),
        ("logreg", LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        )),
    ])

    pipeline.fit(X_train, y_train)

    # 4) evaluate on validation set with default threshold 0.5
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_pred_default = (val_proba >= 0.5).astype(int)

    print("\n=== Validation metrics @ threshold=0.5 ===")
    print(classification_report(y_val, val_pred_default))
    print("Val ROC AUC:", roc_auc_score(y_val, val_proba))

    # 5) cost-aware thresholding
    cost_fp = 1.0   # cost of wrongly targeting a non-churner
    cost_fn = 5.0   # cost of missing a true churner

    best_thr, best_cost = find_best_threshold(y_val.values, val_proba, cost_fn, cost_fp)
    print(f"\nBest cost-aware threshold on val: {best_thr:.3f} (cost={best_cost:.1f})")

    # 6) evaluate on test set with chosen threshold
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)

    print("\n=== Test metrics @ cost-aware threshold ===")
    print(classification_report(y_test, test_pred))
    print("Test ROC AUC:", roc_auc_score(y_test, test_proba))

    # 7) save model + threshold
    model_path = "models/tfidf_logreg_churn.joblib"
    meta_path = "models/threshold.json"

    joblib.dump(pipeline, model_path)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "threshold": float(best_thr),
                "cost_fp": float(cost_fp),
                "cost_fn": float(cost_fn),
            },
            f,
            indent=2,
        )

    print(f"\nSaved model to {model_path}")
    print(f"Saved threshold metadata to {meta_path}")


if __name__ == "__main__":
    main()