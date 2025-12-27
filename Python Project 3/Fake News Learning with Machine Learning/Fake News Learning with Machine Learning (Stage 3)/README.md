# Fake News Learning with Machine Learning — Stage 3

This stage extends Stage 2 by adding **probability calibration** (when needed) and a **threshold sweep** to select a better decision threshold for the “Fake” class.

## Quickstart

```bash
# 1) Train + evaluate + threshold tuning
python src/train.py --data data/fake_and_real_news.csv --text-col Text --label-col label

# 2) Optional: batch score the full dataset (uses chosen_threshold from reports/metrics.json when available)
python src/predict.py   --model artifacts/best_model.joblib   --metrics reports/metrics.json   --data data/fake_and_real_news.csv   --text-col Text   --label-col label   --out reports/preds_scored.csv

# 3) Streamlit dashboard
streamlit run app.py
```

## Mermaid diagram: Stage 3 pipeline

Render this in GitHub (or any Mermaid-enabled Markdown viewer):

```mermaid
flowchart TD
  A[Dataset CSV - data/fake_and_real_news.csv] --> B[Load and validate columns - Text and label]
  B --> C[Train Test split - stratified if possible]
  C --> D[Train candidate models - TFIDF plus LogReg, LinearSVM, MultinomialNB]
  D --> E[Pick best model - macro F1 then F1 Fake then accuracy]
  E --> F{Model supports predict_proba}
  F -- Yes --> G[Use probabilities directly]
  F -- No --> H[Calibrate model using CalibratedClassifierCV]
  G --> I[Threshold sweep - 0.00 to 1.00]
  H --> I
  I --> J[Choose best threshold - chosen_threshold]
  J --> K[Write artifacts - best_model.joblib and label_map.json]
  J --> L[Write reports - metrics.json, threshold_sweep.csv, preds.csv]
  K --> M[predict.py - batch scoring]
  L --> M
  M --> N[reports/preds_scored.csv]
  K --> O[app.py - Streamlit dashboard]
  L --> O
  N --> O
```

## Outputs

Typical outputs after a full Stage 3 run:

- `artifacts/best_model.joblib`
- `artifacts/label_map.json`
- `reports/metrics.json` (includes `chosen_threshold` when generated)
- `reports/threshold_sweep.csv` (threshold vs metrics)
- `reports/model_comparison.csv`
- `reports/preds.csv` / `reports/preds_scored.csv`
