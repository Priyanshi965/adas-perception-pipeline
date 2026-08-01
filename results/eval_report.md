# Crossing-Intent Model — Single-Split Evaluation

- Generated: 2026-08-01T11:46:04
- Command: `python adas_pipeline/evaluate.py --videos 150 --seed 42`
- Videos: 150 · Seed: 42 · Features: POSE+KIN
- Observation window: 16 steps · Prediction horizon (TTE): 15 steps
- Operating threshold: 0.525 (model)
- Samples: 388 (positives 60)

| Metric | Value |
|---|---|
| Accuracy | 0.8067 |
| Precision | 0.4242 |
| Recall | 0.7000 |
| F1 | 0.5283 |
| ROC-AUC | 0.8064 |
| Avg-Precision | 0.4235 |

### Confusion matrix

| | Pred: not-cross | Pred: cross |
|---|---|---|
| **True: not-cross** | 271 | 57 |
| **True: cross** | 18 | 42 |

