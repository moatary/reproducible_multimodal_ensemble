
# Interpretability Sub-Project — Feature Importance Simulations

[![Sub-page: Interpretability Simulations](https://img.shields.io/badge/Sub--page-Interpretability-blue)](#)

Implements and evaluates the **modified FI metric** (Equation 1 of the paper)
against the standard sklearn feature importance.

## Usage

```bash
cd interpretability/
python run_interpretability.py --dataset iemocap --mode global
python run_interpretability.py --dataset iemocap --mode local
```

## FI Metric Variants (Table 6 of paper)

| Variant | Description                                        | Best Avg Score |
|---------|----------------------------------------------------|----------------|
| Spec 5  | Right-branch positive density × node coverage     | **0.72**       |
| Spec 12 | F-beta weighted with positive recall emphasis      | **0.70**       |
| Default | Standard entropy-based impurity reduction (spec 0) | 0.54           |

## Figures

### Global FI — Modified vs Default

![Hierarchical FI comparison](figures/image62.png)
*Modified FI reduces importance of emotionally neutral tokens (time, person, except)*

![Global FI colourisation](figures/image63.png)
*Happy vs sad concept colourisation with modified FI (left) vs default FI (right)*

### Local (Sample-level) FI

![Local FI — sample level](figures/image59.png)
*Each sloping line = one sentence. Thicker/more intense = higher FI.*

![Audio FI happy](figures/image60.png)
![Audio FI sad](figures/image61.png)
*Audio feature importance for happy and sad samples.*
