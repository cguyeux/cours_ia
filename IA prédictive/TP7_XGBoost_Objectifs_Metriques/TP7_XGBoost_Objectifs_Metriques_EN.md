# TP7 - XGBoost: Objectives and Metrics

[Back to contents](../../LISEZMOI.md)

## Learning objectives

- Understand the role of the objective function in XGBoost and choose it based on the problem.
- Identify available objectives for specific cases (Poisson, Tweedie, etc.).
- Choose and combine training, validation, and business-facing metrics.
- Define a custom objective and a custom metric to meet a business constraint.
- Apply these concepts to classification and regression scenarios.

## 1. Role of the objective function

The **objective function** (or loss function) is what XGBoost tries to minimize. It combines:

1. A **training loss** (for example squared error for regression).
2. A **regularization term** (L1/L2) that penalizes tree complexity.

At each iteration, XGBoost adds a tree that reduces this function. The objective choice is therefore crucial: it defines how the model measures error and updates gradients.

### 1.1 Default value by API

| XGBoost API | Estimator | Default objective |
|-------------|-----------|------------------|
| `xgboost.XGBClassifier` | Binary (`y` with 2 classes) | `binary:logistic` (log-loss with sigmoid output)
| `xgboost.XGBClassifier` | Multiclass (`num_class > 2`) | `multi:softprob` (multiclass log-loss with probabilities)
| `xgboost.XGBRegressor`  | Regression | `reg:squarederror` (MSE)
| `xgboost.train` | Booster `gbtree` | `reg:squarederror` if unspecified |

### 1.2 Overview of useful objectives

| Task type | Objective | Description | When to use |
|-----------|-----------|-------------|-------------|
| Binary classification | `binary:logistic` | Binary log-loss, output in [0,1]. | Standard cases: churn, fraud, binary diagnosis. |
| Binary classification | `binary:logitraw` | Binary log-loss with raw output (log-odds). | When you want to control the sigmoid transform or apply a non-standard threshold. |
| Multiclass classification | `multi:softprob` | Multiclass log-loss, outputs probabilities per class. | When you want full probability outputs. |
| Multiclass classification | `multi:softmax` | Multiclass log-loss, outputs predicted class directly. | When only final classes matter (no probabilities). |
| Regression | `reg:squarederror` | Mean squared error. | Classic regression, continuous values. |
| Robust regression | `reg:absoluteerror` | Mean absolute error (MAE). | When outliers are frequent and you want a less sensitive loss. |
| Counts | `count:poisson` | Poisson log-likelihood. | Rare event counts, visit counts, incident counts. |
| Insurance / energy | `reg:tweedie` | Tweedie deviance (between Poisson and Gamma). | Modeling amounts with mass at zero + positive tail (claims). |
| Ranking | `rank:pairwise`, `rank:ndcg`, ... | Ranking-inspired losses. | Recommendation, search engines. |
| Survival | `survival:cox` | Cox model (censored data). | Survival analysis, time-to-event. |

#### Focus on Poisson and Tweedie

- **`count:poisson`**: the target must be a positive integer. XGBoost applies an exponential transform in output, imposing learning on log-counts. It is essential when variance grows with the mean.
- **`reg:tweedie`**: covers a continuum between Poisson (`power=1`), Gamma (`power=2`), and intermediate distributions (`1 < power < 2`). Useful for insurance amounts (many zeros + long tail). Requires `tweedie_variance_power`.

```python
from xgboost import XGBRegressor

model_poisson = XGBRegressor(
    objective="count:poisson",
    max_depth=4,
    tree_method="hist",
)

model_tweedie = XGBRegressor(
    objective="reg:tweedie",
    tweedie_variance_power=1.5,
    max_depth=4,
    tree_method="hist",
)
```

## 2. Choosing the right evaluation metric

The metric (`eval_metric`) tracks performance during training on the training set and validation sets provided in `eval_set`. By default, XGBoost chooses a metric consistent with the objective, but it is often useful to:

1. **Track a business metric** that is interpretable (MAE, F1-score, etc.) on validation to communicate with stakeholders.
2. **Use a metric optimized by XGBoost** (log-loss, RMSE, ...) for early stopping (`early_stopping_rounds`).

### 2.1 Example: imbalanced classification

Suppose a fraud use case where recall is prioritized. We can:

- Optimize the default objective (`binary:logistic`).
- Use `eval_metric=["auc", "logloss"]` to guide training.
- Compute a business metric (for example F1) at each epoch using a callback or post-processing via `model.evals_result_`.

```python
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

model = XGBClassifier(
    scale_pos_weight=10,
    eval_metric=["auc", "logloss"],
    early_stopping_rounds=30,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)

# Business metric on validation
val_preds = model.predict(X_val)
print("Validation F1:", f1_score(y_val, val_preds))
```

### 2.2 Example: regression with logged MAE

```python
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

reg = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    objective="reg:squarederror",
    eval_metric=["rmse"],  # metric used for early stopping
    early_stopping_rounds=30,
)

reg.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)

val_preds = reg.predict(X_val)
print("Validation MAE (business metric):", mean_absolute_error(y_val, val_preds))
```

Here, XGBoost monitors `rmse` (aligned with the MSE objective) for early stopping, while we log `MAE` for decision makers.

### 2.3 Multiple metrics and callbacks

You can provide a **list** to `eval_metric`. XGBoost logs all metrics but uses only the first one for early stopping. After training, `model.evals_result()` returns a dictionary containing all metric curves for each dataset.

For metrics that do not exist natively (for example a business-specific score), you can write a **custom callback** derived from `xgboost.callback.TrainingCallback`:

```python
import numpy as np
import xgboost as xgb

mae_history = []

class LogMAE(xgb.callback.TrainingCallback):
    def __init__(self, dval, y_val):
        self.dval = dval
        self.y_val = y_val

    def after_iteration(self, model, epoch, evals_log):
        preds = model.predict(self.dval)
        mae = float(np.mean(np.abs(preds - self.y_val)))
        mae_history.append(mae)
        print(f"[epoch {epoch}] validation-mae(business)={mae:.4f}")
        return False  # continue training

params = {"objective": "reg:squarederror", "eta": 0.05}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

callbacks = [LogMAE(dval, y_val)]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dtrain, "train"), (dval, "validation")],
    callbacks=callbacks,
)
```

Here, `LogMAE` computes and logs an interpretable metric while XGBoost optimizes its main metric (`rmse`, `logloss`, etc.).

## 3. Custom objectives and metrics

Sometimes a business constraint is not covered by standard objectives. You can then define:

- a **custom objective** that provides gradient and Hessian,
- a **custom metric** to measure performance according to a specific criterion.

### 3.1 General syntax with `xgb.train`

```python
import xgboost as xgb

def custom_objective(preds, dtrain):
    # preds: raw booster outputs (before transformation)
    # dtrain: DMatrix with labels
    labels = dtrain.get_label()
    grad = ...  # first derivative
    hess = ...  # second derivative
    return grad, hess

def custom_metric(preds, dtrain):
    labels = dtrain.get_label()
    metric_value = ...
    return "metric_name", metric_value

params = {
    "max_depth": 4,
    "eta": 0.05,
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    obj=custom_objective,
    feval=custom_metric,
    evals=[(dtrain, "train"), (dval, "validation")],
    early_stopping_rounds=30,
)
```

### 3.2 Example: forcing prediction variance to match target variance

**Objective**: make predictions have the same variance as the target. Define a loss that penalizes the difference between prediction variance and target variance.

Let \( \sigma_y^2 \) be the target variance and \( \sigma_{\hat{y}}^2 \) the prediction variance. We minimize:

\[
\mathcal{L} = \big( \sigma_{\hat{y}}^2 - \sigma_y^2 \big)^2.
\]

The gradient with respect to prediction \(\hat{y}_i\) is:

\[
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{4}{n} \big( \sigma_{\hat{y}}^2 - \sigma_y^2 \big) (\hat{y}_i - \overline{\hat{y}}),
\]

where \( n \) is the number of samples and \(\overline{\hat{y}}\) is the mean prediction.

For the Hessian, use a positive constant approximation to stabilize training.

```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Prepare data
X, y = fetch_california_housing(return_X_y=True, as_frame=False)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

sigma_y2 = np.var(y_train)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

def variance_matching_obj(preds, dtrain):
    labels = dtrain.get_label()
    n = preds.size
    mean_pred = np.mean(preds)
    var_pred = np.mean((preds - mean_pred) ** 2)
    grad = (4.0 / n) * (var_pred - sigma_y2) * (preds - mean_pred)
    hess = np.full_like(preds, 4.0 / n)  # positive approximation
    return grad, hess

def variance_ratio_metric(preds, dtrain):
    labels = dtrain.get_label()
    var_pred = np.var(preds)
    var_label = np.var(labels)
    ratio = var_pred / (var_label + 1e-12)
    return "var_ratio", ratio

params = {
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",  # used to initialize the booster
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=400,
    obj=variance_matching_obj,
    feval=variance_ratio_metric,
    evals=[(dtrain, "train"), (dval, "validation")],
    early_stopping_rounds=30,
)
```

**Interpretation:**

- `variance_matching_obj` pushes prediction variance toward target variance.
- `variance_ratio_metric` logs the ratio \( \sigma_{\hat{y}}^2 / \sigma_y^2 \). Expect a ratio near 1 on validation.
- You can combine this objective with a standard loss via a mixture (for example \( \mathcal{L}_{total} = \text{MSE} + \lambda \mathcal{L}_{variance} \)) by adding the MSE gradient.

### 3.3 Variant: blend with MSE

```python
def blended_objective(preds, dtrain, lam=0.1):
    labels = dtrain.get_label()
    n = preds.size
    mean_pred = np.mean(preds)
    var_pred = np.mean((preds - mean_pred) ** 2)
    # variance component
    grad_var = (4.0 / n) * (var_pred - sigma_y2) * (preds - mean_pred)
    hess_var = np.full_like(preds, 4.0 / n)
    # MSE component
    residuals = preds - labels
    grad_mse = residuals
    hess_mse = np.ones_like(preds)
    grad_total = grad_mse + lam * grad_var
    hess_total = hess_mse + lam * hess_var
    return grad_total, hess_total
```

This keeps good MSE performance while constraining prediction variability.

## 4. Lab exercises

### Exercise 1 - Explore standard objectives

1. Reuse the Breast Cancer Wisconsin dataset (binary classification).
2. Train three models:
   - `objective="binary:logistic"`, `eval_metric=["logloss"]`;
   - `objective="binary:logitraw"`, `eval_metric=["auc", "error"]`;
   - `objective="rank:pairwise"` with `eval_metric=["auc"]` (observe convergence differences).
3. Compare performance (AUC, accuracy) and note effects on probability calibration.

### Exercise 2 - Choose a business metric

1. On a regression problem (California Housing), train an `XGBRegressor` with `objective="reg:squarederror"`.
2. Track `rmse` (for early stopping) and `mae` (business metric) via `eval_metric=["rmse", "mae"]`.
3. Plot the `rmse` and `mae` curves for train and validation and discuss the value of each metric.

### Exercise 3 - Adjust a Poisson or Tweedie objective

1. Build a count dataset (e.g., number of bike rentals per hour) or use open data.
2. Compare objectives `reg:squarederror`, `count:poisson`, and `reg:tweedie` (`tweedie_variance_power=1.3`).
3. Analyze the impact on predictions (positivity, dispersion) and metrics `rmse`, `mae`, `mean_poisson_deviance`.

### Exercise 4 - Implement the "equal variance" objective

1. Follow the example in section 3.2 to define `variance_matching_obj` and `variance_ratio_metric`.
2. Measure the variance ratio on validation and compare to a standard model (`reg:squarederror`).
3. Test different `lam` values in the blended version (section 3.3) to balance accuracy and variability.

### Exercise 5 - Log a custom metric

1. Create a `feval` function that returns MAPE (Mean Absolute Percentage Error) on the validation set.
2. Integrate it into `xgb.train` while keeping `rmse` as the early stopping metric.
3. Export `rmse` and `MAPE` curves and discuss interpretability for a non-technical audience.

## Key takeaways

- The default objective matches the task but can be changed for specific needs (counts, insurance, ranking, etc.).
- You can track multiple metrics simultaneously: one for XGBoost, others for decision makers.
- Custom objectives and metrics let you integrate business constraints (variance, asymmetry, etc.), at the cost of computing gradient and Hessian.
- Mixing a custom objective with a standard loss is often necessary to preserve predictive quality while respecting the constraint.
