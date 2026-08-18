# Crossing-Intent Model — Before/After Comparison (PIE)

- Dataset: PIE (sets set01/02/05), 228 pedestrian tracks (5-fold cross-validation, split at track level)
- Task: crossing-**onset** prediction, observe 16 steps → predict within 15 steps (leak-free)
- Evaluation: 12597 out-of-fold test windows (611 positive), pooled across folds so every track is tested once
- Operating point: all models compared at a common recall of 0.58 (the baseline's); AUC/AP are threshold-free

| Feature set | AUC | AP | Accuracy | BalancedAcc | Precision | Recall | F1 | AUC (mean±std) |
|---|---|---|---|---|---|---|---|---|
| Trajectory only (before) | 0.684 | 0.131 | 0.749 | 0.669 | 0.109 | 0.581 | 0.184 | 0.719 ± 0.111 |
| Body-language only | 0.786 | 0.160 | 0.824 | 0.709 | 0.153 | 0.581 | 0.243 | 0.787 ± 0.070 |
| Pose + trajectory (after) | 0.797 | 0.228 | 0.824 | 0.709 | 0.153 | 0.581 | 0.243 | 0.806 ± 0.078 |

Generated from `compare_models.py --dataset pie` (5-fold CV). Figures in `results_pie/ (cmp_*.png)`.