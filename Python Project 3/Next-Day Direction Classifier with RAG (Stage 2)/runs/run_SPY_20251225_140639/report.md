# Run Report — SPY — 20251225_140639

## Notes
- Time-based split (no shuffling)
- Stage 2: model comparison; select best by **macro-F1**, then accuracy.
- Local learning project; not investment advice.

## Data Snapshots
- Raw saved: `/Users/johndoan/Documents/GitHub/Python/Python Project 3/Next-Day Direction Classifier with RAG (Stage 2)/data/raw/SPY_20251225_140639.csv`
- Processed saved: `/Users/johndoan/Documents/GitHub/Python/Python Project 3/Next-Day Direction Classifier with RAG (Stage 2)/data/processed/SPY_20251225_140639_features.csv`

## Validation Results
- Baseline (majority) — acc: **0.530**, macro-F1: **0.346**

### Candidate Models (VAL)
| model | acc | macro-F1 | F1(up) | F1(down) |
|---|---:|---:|---:|---:|
| `gboost` | 0.525 | 0.471 | 0.639 | 0.303 |
| `rf` | 0.480 | 0.453 | 0.575 | 0.330 |
| `logreg` | 0.527 | 0.401 | 0.676 | 0.127 |

**Selected best model:** `gboost`

## Classification Report (VAL) — Best Model
```text
              precision    recall  f1-score   support

           0       0.49      0.22      0.30       191
           1       0.53      0.80      0.64       215

    accuracy                           0.52       406
   macro avg       0.51      0.51      0.47       406
weighted avg       0.51      0.52      0.48       406
```
