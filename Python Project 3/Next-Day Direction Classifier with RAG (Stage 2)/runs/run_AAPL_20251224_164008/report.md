# Run Report — AAPL — 20251224_164008

## Notes
- Time-based split (no shuffling)
- Stage 2: model comparison; select best by **macro-F1**, then accuracy.
- Local learning project; not investment advice.

## Data Snapshots
- Raw saved: `/Users/johndoan/Documents/GitHub/Python/Python Project 3/Next-Day Direction Classifier with RAG (Stage 2)/data/raw/AAPL_20251224_164008.csv`
- Processed saved: `/Users/johndoan/Documents/GitHub/Python/Python Project 3/Next-Day Direction Classifier with RAG (Stage 2)/data/processed/AAPL_20251224_164008_features.csv`

## Validation Results
- Baseline (majority) — acc: **0.510**, macro-F1: **0.338**

### Candidate Models (VAL)
| model | acc | macro-F1 | F1(up) | F1(down) |
|---|---:|---:|---:|---:|
| `rf` | 0.483 | 0.482 | 0.495 | 0.470 |
| `gboost` | 0.485 | 0.482 | 0.524 | 0.440 |
| `logreg` | 0.498 | 0.405 | 0.640 | 0.171 |

**Selected best model:** `rf`

## Classification Report (VAL) — Best Model
```text
              precision    recall  f1-score   support

           0       0.47      0.47      0.47       199
           1       0.49      0.50      0.50       207

    accuracy                           0.48       406
   macro avg       0.48      0.48      0.48       406
weighted avg       0.48      0.48      0.48       406
```
