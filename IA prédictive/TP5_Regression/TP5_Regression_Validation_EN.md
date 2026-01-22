# TP: Regression, cross-validation, and complexity control

[Back to contents](../../LISEZMOI.md)

## Learning objectives

- Understand the difference between a classification problem and a regression problem.
- Set up supervised regression and interpret its metrics.
- Introduce cross-validation and its role in model selection.
- Explore the impact of the `max_depth` hyperparameter on decision tree complexity.
- Structure an experimental protocol with three sets: train, validation, test.

## Prerequisites

- Python
- `pandas`
- `numpy`
- `matplotlib`
- Basics of `scikit-learn`

## Context

In previous labs you worked on **classification** tasks, where the goal is to predict a discrete label (benign/malignant, yes/no, customer type, etc.). In this lab, we focus on **regression**: the model must predict a continuous value (price, temperature, consumption, duration, ...).

We will use two datasets:

1. **California Housing** (scikit-learn): predicting median housing prices per district.
2. **Synthetic DataFrame** with a non-linear relationship between variables.

## Part 1 - Classification vs Regression

### 1.1 Theoretical comparison

- **Classification**: discrete output, metrics such as accuracy, F1-score.
- **Regression**: continuous output, metrics such as MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), $R^2$.

**Question 1.** Based on your experience in previous labs, explain when you would choose classification or regression. Illustrate with a business example.

### 1.2 First steps in regression

In a Jupyter notebook, implement the following code to load the California Housing dataset and train a `DecisionTreeRegressor`.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load the dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Regression model
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)

# Evaluation
preds = reg.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

print(f"MAE : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2 : {r2:.3f}")
```

**Question 2.** Add a cell that converts the metrics into a small DataFrame, then comment on their meaning.

## Part 2 - Cross-validation and data splits

### 2.1 Why go beyond a simple train/test split?

When you add new features or tune hyperparameters, you can improve performance **by chance** on the test set. This means your model adapts to this specific test set, but does not necessarily generalize. To avoid this:

- **Train set**: used to fit the model parameters.
- **Validation set**: used during training to decide when to stop and avoid overfitting.
- **Test set**: used to compare models/hyperparameters and choose the best configuration.

### 2.2 $k$-fold cross-validation

Cross-validation splits the training set into $k$ subsets. Each time, you use $k-1$ subsets to train and the remaining subset to validate. You repeat this $k$ times and average the metrics.

**Code to complete:**

```python
from sklearn.model_selection import KFold, cross_val_score

reg = DecisionTreeRegressor(max_depth=5, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    reg,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=kf
)

print("RMSE scores (negative):", scores)
print("Average RMSE:", -scores.mean())
print("Std dev:", scores.std())
```

**Question 3.** Explain why scikit-learn returns negative scores for RMSE. How do you convert these values to interpret them?

**Question 4.** Compare the mean RMSE obtained with cross-validation to the one computed on the test set without CV. What do you observe?

### 2.3 Model selection via cross-validation

We will compare three models:

- `DecisionTreeRegressor`
- `RandomForestRegressor`
- `GradientBoostingRegressor`

**Guided exercise:**

1. Create a function `evaluate_model(model, X, y)` that returns the mean RMSE in 5-fold cross-validation.
2. Apply it to the three models with their default parameters.
3. Store results in a comparative DataFrame.
4. Select the model with the best mean RMSE and retrain it on the full training set.
5. Evaluate on the test set and comment on the consistency between cross-validation and test.

## Part 3 - Focus on `max_depth`

### 3.1 Understanding tree complexity

The `max_depth` hyperparameter limits the number of levels in a decision tree. A tree that is too deep memorizes the data (overfitting), while a tree that is too shallow fails to capture structure (underfitting).

**Exercise:**

```python
results = []
for depth in range(2, 11):
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    rmse_cv = -cross_val_score(
        reg,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=5
    ).mean()
    results.append({"max_depth": depth, "rmse_cv": rmse_cv})

results_df = pd.DataFrame(results)
print(results_df)
```

**Question 5.** Plot `max_depth` vs `rmse_cv`. From which depth does RMSE stabilize?

### 3.2 Visualize on a synthetic dataset

Create an artificial dataset with a non-linear relationship (e.g., `y = sin(x) + noise`). Use `DecisionTreeRegressor` with increasing `max_depth` to visually observe underfitting and overfitting.

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)
X_syn = np.sort(5 * rng.rand(200, 1), axis=0)
y_syn = np.sin(X_syn).ravel()
y_syn += 0.5 * (rng.rand(200) - 0.5)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, depth in zip(axes.ravel(), [2, 4, 6, 10]):
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    reg.fit(X_syn, y_syn)
    ax.scatter(X_syn, y_syn, s=10, label="Data")
    ax.plot(X_syn, reg.predict(X_syn), color="red", label="Prediction")
    ax.set_title(f"max_depth = {depth}")
    ax.legend()

plt.tight_layout()
plt.show()
```

**Question 6.** Comment on the curves obtained. What bias/variance tradeoff do you observe?

### 3.3 Best practices

- Always test different `max_depth` values (e.g., 2 to 10) to calibrate complexity.
- Monitor RMSE (or another metric) on the validation set: if performance degrades after a certain depth, it indicates overfitting.
- Use cross-validation to make the choice robust: a single train/test split can be misleading.

## Part 4 - Synthesis and extensions

1. Write a short paragraph explaining how regression differs from classification, mentioning metrics, risks, and tools used in this lab.
2. Explain how cross-validation helps you decide whether to add a new feature or change a hyperparameter.
3. Describe your strategy to determine the right `max_depth` value on a new project.

## Additional resources

- scikit-learn documentation - [Regression](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- Towards Data Science article - [Understanding Cross-Validation](https://towardsdatascience.com/cross-validation-70289113a072)
- scikit-learn documentation - [`DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

**Expected deliverable:** a Jupyter notebook containing the code, answers to the questions, and a concise conclusion about modeling choices.
