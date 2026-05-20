
# Fine-Tuning Sub-Project

Comprehensive hyperparameter search that produced the optimal configurations
reported in **Table 1** of the paper.

## How Optimal Hyperparameters Were Obtained

The search follows **sequential coordinate-wise optimisation**:

1. Fix all hyperparameters at defaults.
2. Sweep one axis while holding others fixed; select the value maximising F1.
3. Update the default; proceed to the next axis.
4. Repeat the full pass until convergence.

## Optimal Hyperparameters (Table 1 of the paper)

| Classifier | Hyperparameter        | Search Space               | **Best Value** |
|------------|-----------------------|----------------------------|----------------|
| DT         | Max depth             | {5, 10, 20, None}          | **20**         |
| DT         | Min samples split     | {2, 5, 10}                 | **5**          |
| DT         | Criterion             | {gini, entropy}            | **entropy**    |
| RF         | N estimators          | {50, 100, 200, 500}        | **200**        |
| RF         | Max depth             | {10, 20, None}             | **None**       |
| RF         | Max features          | {sqrt, log2, 0.5}          | **sqrt**       |
| RF         | Min samples leaf      | {1, 2, 4}                  | **1**          |
| AdaBoost   | N estimators          | {50, 100, 300}             | **100**        |
| AdaBoost   | Learning rate         | {0.01, 0.1, 0.5, 1.0}     | **0.1**        |
| AdaBoost   | Base estimator depth  | {1, 2, 3}                  | **2**          |
| AdaBoost   | Algorithm             | {SAMME, SAMME.R}           | **SAMME.R**    |
| XGBoost    | N estimators          | {100, 300, 500}            | **300**        |
| XGBoost    | Max depth             | {3, 6, 9}                  | **6**          |
| XGBoost    | Learning rate         | {0.01, 0.05, 0.1}          | **0.05**       |
| XGBoost    | Subsample             | {0.6, 0.8, 1.0}            | **0.8**        |
| XGBoost    | Col sample by tree    | {0.6, 0.8, 1.0}            | **0.8**        |
| LDT/LDF    | LDA components        | {1, 2, 5, 10}              | **5**          |
| LDT/LDF    | LDA shrinkage         | {None, 0.1, 0.5, auto}     | **auto**       |
| LDT/LDF    | Base tree depth       | {5, 10, 20}                | **10**         |

## Usage

```bash
cd finetuning/
python run_finetuning.py --dataset iemocap --criterion f1 --seed 42
```
Results written to `results/tuning_log.txt`.
