
# Classification Sub-Project — Accuracy Simulations

[![Sub-page: Accuracy Simulations](https://img.shields.io/badge/Sub--page-Accuracy%20Simulations-brightgreen)](#)

Each classifier is run with its **paper-optimal hyperparameters** (Table 1),
averaged over 10 independent seeds for reproducibility.

## Usage

```bash
cd classification/
python random_forest/run_rf.py  --dataset iemocap
python decision_tree/run_dt.py  --dataset iemocap
python xgboost/run_xgb.py       --dataset iemocap
python adaboost/run_ab.py       --dataset iemocap
python ldt/run_ldt.py           --dataset iemocap
python ldf/run_ldf.py           --dataset iemocap
```

## Results (IEMOCAP, 10-seed average)

| Model       | Accuracy | Weighted F1 | F1-mod (Happy) | F1-mod (Sad) |
|-------------|----------|-------------|----------------|--------------|
| DT          | 65.3     | 63.1        | 61.2           | 59.8         |
| RF ★        | **74.1** | **72.3**    | 68.1           | 67.4         |
| AdaBoost    | 70.2     | 68.9        | 65.3           | 63.7         |
| XGBoost     | 72.8     | 71.4        | 67.5           | 65.9         |
| LDT         | 71.5     | 69.8        | 66.2           | 64.8         |
| LDF ★       | 73.4     | 71.8        | **69.2**       | **68.1**     |

★ Best two results bolded; `^^` = statistically significant (p<0.05 vs IMR baseline).

## Figures

![Elite performance frequency](figures/image77.png)
*Frequency of elite performance across classifiers.*

![Accuracy vs n-gram count](figures/image76.png)
*Ablation: accuracy as a function of n-gram cluster count.*
