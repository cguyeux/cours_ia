# TP8 Path - Preprocessing and feature selection

[Back to contents](../../LISEZMOI.md)

This path is split into **four complementary labs**, each focused on a key preprocessing step. They can be completed independently, but it is recommended to follow them in order to build a robust pipeline.

| Lab | Main topic | Skills practiced |
| --- | --- | --- |
| [TP8.1 - Feature selection](TP8_1_Selection_Variables_EN.md) | Understand and compare feature selection techniques. | Statistical analysis, cross-validation, model interpretation. |
| [TP8.2 - Scaling and encoding](TP8_2_Normalisation_Encodage_EN.md) | Build transformation pipelines suited to variable types. | Transformer choice, leakage prevention, model comparison. |
| [TP8.3 - Outlier detection and handling](TP8_3_Valeurs_Aberrantes_EN.md) | Identify and handle outliers before training. | Statistical methods, automated detection, model robustness. |
| [TP8.4 - Imbalanced classification](TP8_4_Classification_Desequilibree_EN.md) | Adapt classification models to imbalanced datasets. | Resampling strategies, suitable metrics, algorithm tuning. |

## Cross-cutting learning objectives

- Understand the key steps of feature preparation before model training.
- Select relevant features and reduce dimensionality.
- Apply scaling to quantitative variables and encoding to categorical variables.
- Identify and handle outliers.
- Adapt classification pipelines to imbalanced datasets.
- Discuss specific impacts on algorithms such as XGBoost.

## Prerequisites

- Python, `pandas`, `numpy`, `matplotlib`, `scikit-learn`.
- Basic knowledge of cross-validation, pipelines, and model evaluation (accuracy, f1-score, ROC/AUC, etc.).

## Recommended dataset

All labs use the **Adult Income** dataset (UCI Census Income) available via `fetch_openml`. The goal is to predict whether annual income exceeds 50K using socio-demographic variables (numeric and categorical) with moderate imbalance.

```python
from sklearn.datasets import fetch_openml
import pandas as pd

adult = fetch_openml(name="adult", version=2, as_frame=True)
X = adult.data
y = adult.target
```

You can use another tabular dataset if you justify it in your deliverables.

## Expected deliverables

For each lab, provide:

1. A documented Jupyter notebook with code, visualizations, and answers.
2. A 5-10 line synthesis highlighting key takeaways (choices, limitations, improvement ideas).
3. The chosen hyperparameters and the relevant metrics used for comparison.

A global synthesis (10-15 lines) is required at the end of the path to connect your conclusions.
