# TP4 - Advanced XGBoost and validation set handling

[Back to contents](../../LISEZMOI.md)

## Learning objectives

- Understand the value of a validation set when training an XGBoost model.
- Explain the role of `learning_rate` and know when to adjust it.
- Implement performance monitoring with `eval_set` and early stopping.
- Tune hyperparameters (notably `learning_rate` and `n_estimators`) to benefit from early stopping.
- Apply these concepts to the Breast Cancer Wisconsin dataset used in the previous lab.

## Understanding `learning_rate`

The `learning_rate` (also called **eta**) determines the gradient step size at each iteration. The smaller it is, the more each new tree only slightly corrects the errors of previous trees. Conversely, a higher rate makes larger jumps in the solution space.

The default XGBoost value is often satisfactory: it is a good tradeoff between learning speed and stability. Two common situations justify an adjustment:

1. **Convergence is too slow**: if, by inspecting the logs (`verbose=True`), the metric improves very slowly or the model needs many iterations to stabilize, it means each iteration learns too little. You can **slightly increase `learning_rate`** to speed convergence.
2. **Very fast stop with no gain**: if training ends almost immediately, for example by hitting `early_stopping_rounds` (15 iterations in the example), updates are too abrupt. The model cannot improve beyond the first iterations. You should **reduce `learning_rate`** to take finer steps.

Adjusting `learning_rate` always goes with the number of trees (`n_estimators`). A lower rate requires more trees to converge, while a higher rate needs fewer.

## Why use a validation set?

When training a model, it is tempting to evaluate performance only on the training set. Yet this often leads to **overfitting**: the model memorizes the training data and generalizes poorly to new data.

A validation set allows you to monitor model performance on data **never seen during parameter tuning**. For XGBoost, this means tracking a metric (log loss, error, AUC, etc.) on the validation set and stopping training when it no longer improves.

### Key benefits

- **Early detection of overfitting**: as soon as validation performance degrades, training stops.
- **Time savings**: no need to reach the maximum number of trees if quality no longer improves.
- **Automatic best model selection**: XGBoost keeps the weights from the best validation iteration.

## Practical setup in XGBoost

Typical steps to introduce a validation set with the `XGBClassifier` (scikit-learn API):

1. **Split the data**:
   - First split into train/test for final evaluation.
   - Then create a validation subset from the train split (for example 80% train / 20% validation).
2. **Choose a coherent `learning_rate` / `n_estimators` pair**:
   - Starting with the default `learning_rate` (`0.1`) is a good practice. Decrease it if learning is unstable or too fast, increase it if it progresses too slowly.
   - Early stopping only works if it has enough trees to scan. For low rates, push `n_estimators` between 800 and 2000 to leave room.
3. **Call `fit` with `eval_set` and `early_stopping_rounds`**:
   - `eval_set=[(X_train, y_train), (X_val, y_val)]` allows monitoring multiple datasets.
   - `early_stopping_rounds=20` stops training if the metric does not improve for 20 consecutive iterations.
   - Watching logs (`verbose=True`) helps judge convergence speed and adapt `learning_rate`.
4. **Choose the metric**:
   - Via `eval_metric` (`logloss`, `auc`, `error`, ...).

### Full code example

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Example with the Breast Cancer Wisconsin dataset (scikit-learn)
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target

# 1. train/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# 3. define a coherent learning_rate / n_estimators pair
model = XGBClassifier(
    n_estimators=1000,      # large value
    learning_rate=0.05,     # smaller than default for finer steps
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="binary:logistic"
)

# 4. training with validation monitoring
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric="logloss",
    early_stopping_rounds=20,
    verbose=True
)

# 5. final evaluation on the test set
preds = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, preds))
print("Best number of trees:", model.best_iteration + 1)
```

Key points to remember:

- **`best_iteration`** provides the tree index (0-based) at the best validation score.
- **`n_estimators` must be larger than `best_iteration`**; if you set it too low (e.g., 50 trees), early stopping can never trigger even if the model is not optimal.
- You can save training metrics via `evals_result = model.evals_result()` to plot the curves.

### Plotting the learning curve

```python
evals_result = model.evals_result()
train_logloss = evals_result["validation_0"]["logloss"]
val_logloss = evals_result["validation_1"]["logloss"]

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(train_logloss, label="Train logloss")
plt.plot(val_logloss, label="Validation logloss")
plt.axvline(model.best_iteration, color="r", linestyle="--", label="Best iteration")
plt.xlabel("Number of trees")
plt.ylabel("Logloss")
plt.title("XGBoost learning curves")
plt.legend()
plt.show()
```

This visualization shows when validation stops improving.

## Additional best practices

- **Set `early_stopping_rounds`** between 10 and 50 depending on dataset size.
- **Keep an independent test set** that is never used for early stopping.
- **Test multiple metrics**: for imbalanced classes, `auc` or `logloss` are more informative than accuracy.
- **Adapt `learning_rate` to training dynamics**: increase it slightly if convergence is too slow, decrease it if training stops too early with no gain.
- **Lower `learning_rate`** when increasing `n_estimators` to obtain finer optimization.

## Lab work

In this lab, reuse the **Breast Cancer Wisconsin** dataset introduced in TP3.

1. Set up a train/validation/test split and train an `XGBClassifier` with an initial pair `(learning_rate=0.1, n_estimators=800)` and `early_stopping_rounds=20`. Observe convergence speed with `verbose`.
2. Then adjust `learning_rate` in both directions:
   - increase it (e.g., `0.2`) to speed convergence if it is too slow; note the effect on the number of iterations before stopping;
   - decrease it (e.g., `0.03`) to avoid premature stopping if training ends too quickly without improvement.
3. Plot the tracked metric curve for the different settings and relate your observations to the chosen parameters (`learning_rate`, `early_stopping_rounds`).
4. Compare test set performance across configurations and with a run without a validation set. What differences do you observe?

Record your observations (best iteration, metric evolution, conclusions about the impact of the validation set) in your report.
