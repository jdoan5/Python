# Run Report — VOO — 20251225_140640

## Notes
- Time-based split (no shuffling)
- Stage 2: model comparison; select best by **macro-F1**, then accuracy.
- Local learning project; not investment advice.

## Data Snapshots
- Raw saved: `/Users/johndoan/Documents/GitHub/Python/Python Project 3/Next-Day Direction Classifier with RAG (Stage 2)/data/raw/VOO_20251225_140640.csv`
- Processed saved: `/Users/johndoan/Documents/GitHub/Python/Python Project 3/Next-Day Direction Classifier with RAG (Stage 2)/data/processed/VOO_20251225_140640_features.csv`

## Validation Results
- Baseline (majority) — acc: **0.532**, macro-F1: **0.347**

### Candidate Models (VAL)
| model | acc | macro-F1 | F1(up) | F1(down) |
|---|---:|---:|---:|---:|
| `gboost` | 0.527 | 0.473 | 0.642 | 0.304 |
| `rf` | 0.480 | 0.446 | 0.584 | 0.308 |
| `logreg` | 0.527 | 0.421 | 0.669 | 0.172 |

**Selected best model:** `gboost`

## Classification Report (VAL) — Best Model
```text
              precision    recall  f1-score   support

           0       0.49      0.22      0.30       190
           1       0.54      0.80      0.64       216

    accuracy                           0.53       406
   macro avg       0.51      0.51      0.47       406
weighted avg       0.51      0.53      0.48       406
```
