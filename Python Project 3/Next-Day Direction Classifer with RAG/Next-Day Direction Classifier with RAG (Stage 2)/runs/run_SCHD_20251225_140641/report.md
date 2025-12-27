# Run Report — SCHD — 20251225_140641

## Notes
- Time-based split (no shuffling)
- Stage 2: model comparison; select best by **macro-F1**, then accuracy.
- Local learning project; not investment advice.

## Data Snapshots
- Raw saved: `/Users/johndoan/Documents/GitHub/Python/Python Project 3/Next-Day Direction Classifier with RAG (Stage 2)/data/raw/SCHD_20251225_140641.csv`
- Processed saved: `/Users/johndoan/Documents/GitHub/Python/Python Project 3/Next-Day Direction Classifier with RAG (Stage 2)/data/processed/SCHD_20251225_140641_features.csv`

## Validation Results
- Baseline (majority) — acc: **0.530**, macro-F1: **0.346**

### Candidate Models (VAL)
| model | acc | macro-F1 | F1(up) | F1(down) |
|---|---:|---:|---:|---:|
| `rf` | 0.493 | 0.477 | 0.567 | 0.387 |
| `gboost` | 0.461 | 0.460 | 0.480 | 0.440 |
| `logreg` | 0.534 | 0.440 | 0.670 | 0.209 |

**Selected best model:** `rf`

## Classification Report (VAL) — Best Model
```text
              precision    recall  f1-score   support

           0       0.45      0.34      0.39       191
           1       0.52      0.63      0.57       215

    accuracy                           0.49       406
   macro avg       0.48      0.48      0.48       406
weighted avg       0.48      0.49      0.48       406
```
