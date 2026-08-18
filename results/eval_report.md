# Crossing-Intent Model — Single-Split Evaluation

- Generated: 2026-08-01T18:37:13
- Command: `python adas_pipeline/evaluate.py --videos 40 --seed 42`
- Videos: 40 · Seed: 42 · Features: POSE+KIN
- Observation window: 16 steps · Prediction horizon (TTE): 15 steps
- Operating threshold: 0.525 (model)
- Samples: 98 (positives 15)

| Metric | Value |
|---|---|
| Accuracy | 0.7653 |
| Precision | 0.3947 |
| Recall | 1.0000 |
| F1 | 0.5660 |
| ROC-AUC | 0.8506 |
| Avg-Precision | 0.3786 |

### Confusion matrix

| | Pred: not-cross | Pred: cross |
|---|---|---|
| **True: not-cross** | 60 | 23 |
| **True: cross** | 0 | 15 |

