# Crossing-Intent Model (POSE+KIN, PIE) — Single-Split Evaluation

- Generated: 2026-08-11T19:24:51
- Command: `python adas_pipeline/evaluate.py --dataset pie --seed 42`
- Dataset: PIE (sets set01/02/05) · Seed: 42 · Features: POSE+KIN
- Observation window: 16 steps · Prediction horizon (TTE): 15 steps
- Operating threshold: 0.242 (model)
- Samples: 1564 (positives 109)

| Metric | Value |
|---|---|
| Accuracy | 0.5678 |
| Precision | 0.1294 |
| Recall | 0.9083 |
| F1 | 0.2265 |
| ROC-AUC | 0.8445 |
| Avg-Precision | 0.3311 |

### Confusion matrix

| | Pred: not-cross | Pred: cross |
|---|---|---|
| **True: not-cross** | 789 | 666 |
| **True: cross** | 10 | 99 |

