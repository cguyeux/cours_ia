# TP6.4 - Supervised approaches and final comparison

[Back to index](TP6_Series_Temporelles_Index_EN.md)

## Learning objectives

- Turn a series into a supervised dataset (lagged features, calendar features, moving averages).
- Use rolling validation (`TimeSeriesSplit`) to evaluate non-serial models.
- Compare random forests, gradient boosting/XGBoost, and ARIMA.
- Discuss pros/cons of ML approaches for time series forecasting.

## Step 1 - Feature construction

```python
import pandas as pd
import numpy as np

df_features = serie.to_frame(name="count")
df_features["hour"] = df_features.index.hour
df_features["dayofweek"] = df_features.index.dayofweek
df_features["is_weekend"] = df_features["dayofweek"].isin([5, 6]).astype(int)
df_features["month"] = df_features.index.month

for lag in [1, 2, 3, 12, 24, 168]:
    df_features[f"lag_{lag}"] = df_features["count"].shift(lag)

df_features["rolling_mean_4"] = df_features["count"].shift(1).rolling(window=4).mean()
df_features["rolling_mean_24"] = df_features["count"].shift(1).rolling(window=24).mean()
df_features["rolling_std_24"] = df_features["count"].shift(1).rolling(window=24).std()

df_features = df_features.dropna()

X = df_features.drop(columns=["count"])
y = df_features["count"]
```

Tip: add weather data (`temp`, `humidity`, ...) to test the value of exogenous information.

## Step 2 - Time split

- Keep the same split as in TP6.3 (`train up to end of Sep 2012`, `test = last quarter`).
- Use `TimeSeriesSplit` to validate hyperparameters.

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

split_index = X.index < split_date
X_train, X_test = X.loc[split_index], X.loc[~split_index]
y_train, y_test = y.loc[split_index], y.loc[~split_index]

tscv = TimeSeriesSplit(n_splits=5)

def evaluate_model(model):
    maes, rmses = [], []
    for train_idx, val_idx in tscv.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        pred = model.predict(X_train.iloc[val_idx])
        maes.append(mean_absolute_error(y_train.iloc[val_idx], pred))
        rmses.append(mean_squared_error(y_train.iloc[val_idx], pred, squared=False))
    return np.mean(maes), np.mean(rmses)
```

## Step 3 - Supervised models

### 3.1 Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42
)
mae_rf, rmse_rf = evaluate_model(rf)
print(f"RF validation - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
rmse_rf_test = mean_squared_error(y_test, pred_rf, squared=False)
```

### 3.2 XGBoost (or Gradient Boosting)

If `xgboost` is not available, replace with `GradientBoostingRegressor`.

```python
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

mae_xgb, rmse_xgb = evaluate_model(xgb)
print(f"XGB validation - MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")

xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
rmse_xgb_test = mean_squared_error(y_test, pred_xgb, squared=False)
```

### 3.3 Linear baseline (optional)

Testing `Ridge` or `Lasso` helps check whether lags suffice without a nonlinear model.

## Step 4 - Result analysis

### 4.1 Summary table

| Model | MAE validation | RMSE validation | RMSE test |
| --- | --- | --- | --- |
| Persistence (TP6.3) | ... | ... | ... |
| ARIMA (TP6.3) | ... | ... | ... |
| RandomForest | `mae_rf` | `rmse_rf` | `rmse_rf_test` |
| XGBoost / GBoost | `mae_xgb` | `rmse_xgb` | `rmse_xgb_test` |

### 4.2 Visualizations

- Plot `y_test` vs `pred_rf` and `pred_xgb` over 7 days.
- Plot feature importances (RF or XGBoost):

```python
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).head(15).plot(kind="barh", figsize=(8,6))
```

- Analyze dominant features (lags, moving averages, hours).

### 4.3 Multi-horizon forecasts

- Implement a **rolling forecast**: at each step, add the predicted value into the features (for horizon > 1).
- Compare error drift with the ARIMA model.

## Step 5 - Critical discussion

- Do ML models outperform ARIMA? On which aspects (peaks, troughs, smoothing)?
- What is the cost in terms of interpretability and maintenance (feature recomputation)?
- Which exogenous data would you add (weather, school calendar, events)?

## Final deliverables (full path)

1. Summary notebook (or report) including:
   - transformation diagram (TP6.1 -> TP6.4),
   - comparative metrics table,
   - key plots (decomposition, ACF, forecast vs actual, importances).
2. Operational recommendations:
   - expected daily fleet size,
   - alert procedures (extreme values / anomalies),
   - improvement ideas (exogenous data collection, retraining).

Well done, you now have a full time series forecasting pipeline, from exploration to deployment.
