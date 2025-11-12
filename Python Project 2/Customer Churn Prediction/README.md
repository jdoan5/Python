# Customer Churn Prediction

A compact, **diagram‑first** project for training and serving a churn classifier.  
This README is ready to paste into your repository and renders diagrams on GitHub (Mermaid).

> **Tech highlights:** Python, scikit‑learn / XGBoost or LightGBM, SHAP, permutation importance, reproducible runs.

---

## Quick Start

> Requires Python 3.9+

```bash
# 1) (optional) create & activate a venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) install common deps
pip install -U pandas numpy scikit-learn xgboost lightgbm shap matplotlib joblib

# 3) run a baseline
python baseline.py --data data.csv

# 4) train a model
python train.py --data data.csv --model-out models/model.pkl --encoders-out models/encoders.pkl --run-dir runs/exp_001

# 5) explainability (SHAP + permutation importance)
python shap_explain.py --model models/model.pkl --encoders models/encoders.pkl --data data.csv --out runs/exp_001/shap/
python perm_importance.py --model models/model.pkl --encoders models/encoders.pkl --data data.csv --out runs/exp_001/perm/

# 6) predict
python predict.py --model models/model.pkl --encoders models/encoders.pkl --json '{"tenure_months": 12, "monthly_fee": 69.0, "auto_pay": true, "support_tickets_90d": 2, "segment":"consumer"}'
```

> Tip: Replace `xgboost` with `lightgbm` if preferred; scripts are agnostic as long as the estimator implements `.fit` and `.predict_proba`.

---

## Repository Layout

```
.
├── data.csv
├── baseline.py
├── train.py
├── predict.py
├── shap_explain.py
├── perm_importance.py
├── models/
└── runs/
```

---

## System Overview

```mermaid
flowchart LR
  subgraph Client
    U[User/App]
  end

  subgraph API["FastAPI (optional)"]
    P[/POST /predict/]
    V[Validate & Parse]
    FT[Feature Transform]
  end

  subgraph ModelSvc["Model Runtime"]
    M[(Classifier)]
    T{Threshold τ}
  end

  subgraph Infra
    FS[(Encoders / Feature Store)]
    MR[(Model Registry)]
    LG[(Logs & Metrics)]
  end

  U --> P --> V --> FT --> M --> T
  FT <---> FS
  M <---> MR
  T -->|churn| C1[Return: CHURN=1, prob]
  T -->|retain| C0[Return: CHURN=0, prob]
  P --> LG
  M --> LG
```

---

## Data Lineage & Training Pipeline

```mermaid
flowchart TB
  SRC[Raw CSV/Parquet] --> CLN[Clean & Impute]
  CLN --> FE[Feature Engineering<br/> (encoders, bins, ratios)]
  FE --> SPLIT{Train/Valid/Test}
  SPLIT --> TRN[Train Model<br/>(XGB/LGBM/SKLearn)]
  TRN --> EVAL[Evaluate<br/>(AUC/PR, F1, Recall@K)]
  TRN --> EXPL[Explainability<br/>(SHAP, PermImport)]
  EVAL --> ART[Persist Artifacts]
  EXPL --> ART
  ART --> REG[(Register Model + Encoders)]
```

---

## Serving Flow (Request → Decision)

```mermaid
sequenceDiagram
  participant Client
  participant API as FastAPI /predict (optional)
  participant X as Transformer
  participant M as Model
  Client->>API: JSON payload
  API->>X: validate + transform
  X->>M: ndarray(features)
  M-->>API: probability p(churn)
  API-->>Client: {prob: p, label: (p>=τ)}
```

---

## Data Model (Simplified)

```mermaid
erDiagram
  CUSTOMER ||--o{ ACCOUNT : has
  CUSTOMER {
    string customer_id PK
    date   signup_date
    string segment
    bool   is_active
    int    tenure_months
  }
  ACCOUNT {
    string account_id PK
    string customer_id FK
    string plan_type
    float  monthly_fee
    bool   auto_pay
  }
  USAGE }o--|| CUSTOMER : belongs_to
  USAGE {
    string customer_id FK
    int    calls_last_30d
    float  data_gb_last_30d
    int    support_tickets_90d
    bool   promo_used
    bool   churn_label
  }
```

---

## Experiment Lifecycle

```mermaid
stateDiagram-v2
  [*] --> DraftConfig
  DraftConfig --> TrainRun : lock seed & params
  TrainRun --> Evaluate : metrics logged
  Evaluate --> Explain : SHAP, PermImport
  Explain --> Register : save model+preproc
  Register --> Release : tag version
  Release --> [*]
```

---

## Config at a Glance (Optional)

```mermaid
flowchart TB
  CFG[config.yaml]
  CFG --> HP[hyperparameters]
  CFG --> FE2[feature_list]
  CFG --> THR[decision_threshold τ]
  CFG --> PATHS[artifact_paths]
```

---

## Quality Gates

```mermaid
flowchart LR
  M[Candidate Model] --> TESTS[Unit & Data Tests]
  TESTS --> BIAS[Drift & Bias Checks]
  BIAS --> OK{Meets Gates?}
  OK -- Yes --> REG[(Registry)]
  OK -- No --> REWORK[Revise features/params]
```

---

## Rollback Strategy

```mermaid
flowchart LR
  LIVE[(Prod Model vN)] --> MON[Monitor]
  MON --> BAD{Degradation?}
  BAD -- Yes --> RB[Rollback to vN-1]
  BAD -- No --> KEEP[Keep vN]
```

---

## CLI (Arguments Summary)

- `baseline.py` — quick sanity model & metrics on `data.csv`
- `train.py` — trains, evaluates, saves `models/model.pkl` and `models/encoders.pkl`
- `predict.py` — loads artifacts and scores single JSON or a CSV
- `shap_explain.py` — SHAP global & local plots
- `perm_importance.py` — permutation feature importance

> All scripts accept `--help` for exact flags.

---

## Rendering Mermaid Locally

GitHub renders Mermaid diagrams automatically.  
For local preview, use VS Code with **Markdown Preview Mermaid Support** extension.

---

## License

MIT (or your preferred license)
