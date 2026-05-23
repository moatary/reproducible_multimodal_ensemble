
# Interpretable Multimodal Classification with Linear Discriminant Tree Ensembles

[![arXiv](https://img.shields.io/badge/arXiv-2501.48291-b31b1b.svg)](https://arxiv.org/abs/2501.48291)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

> **If you use this code, please cite:**
> ```bibtex
> @article{moattari2025interpretable,
>   title   = {Interpretable Multimodal Classification with Linear Discriminant Tree Ensembles},
>   author  = {Moattari, Mojtaba},
>   year    = {2026},
>   note    = {Preprint: arXiv}
> }
> ```

## Structure

```
reproducible_multimodal_ensemble/
├── finetuning/       # Hyperparameter search (how optimal values were found)
│   ├── model/
│   ├── utils/
│   ├── results/
│   ├── figures/
│   ├── run_finetuning.py
│   └── README.md
├── classification/   # Accuracy Simulations — per-algorithm runs
│   ├── model/
│   ├── utils/
│   ├── figures/
│   ├── decision_tree/
│   ├── random_forest/
│   ├── adaboost/
│   ├── xgboost/
│   ├── ldt/
│   ├── ldf/
│   └── README.md
├── interpretability/ # Interpretability Simulations
│   ├── model/
│   ├── utils/
│   ├── figures/
│   ├── run_interpretability.py
│   └── README.md
├── data/             # Place iemocap_data.pkl, cmu_mosi_data.pkl here
├── requirements.txt
└── README.md
```

## Sub-Project Pages

| Sub-project | Description |
|-------------|-------------|
| [Accuracy Simulations](classification/README.md) | Per-classifier runs with paper-optimal hyperparameters |
| [Interpretability Simulations](interpretability/README.md) | Modified vs default FI analysis |
| [Hyperparameter Fine-Tuning](finetuning/README.md) | Full grid search log |

## Quick Start

```bash
git clone https://github.com/moatary/reproducible_multimodal_ensemble.git
cd reproducible_multimodal_ensemble
pip install -r requirements.txt
# place data files in data/
python classification/random_forest/run_rf.py --dataset iemocap
```

## Results Summary (IEMOCAP, averaged over 10 seeds)

| Model | Accuracy | F1 (weighted) | F1-mod |
|-------|----------|---------------|--------|
| RF    | **74.1** | **72.3**      | 68.1   |
| LDF   | 73.4     | 71.8          | **69.2** |
| XGB   | 72.8     | 71.4          | 67.5   |
| AdaBoost | 70.2  | 68.9          | 65.3   |
| LDT   | 71.5     | 69.8          | 66.2   |
| DT    | 65.3     | 63.1          | 61.2   |

## License

MIT — see [LICENSE](LICENSE).
