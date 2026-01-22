# TP6.3 - AR and ARIMA models

[Back to index](TP6_Series_Temporelles_Index_EN.md)

## Learning objectives

- Build a **persistence baseline** to measure the gains of advanced models.
- Train and interpret an **autoregressive AR(p)** model.
- Configure an **ARIMA(p,d,q)** model consistent with stationarity diagnostics.
- Evaluate models on short-term forecasts and analyze residuals.

## Preparation

- Use the transformed series from TP6.2 (differencing if needed).
- Define a time split: for example, **train = data up to Sep 30, 2012**, **test = last quarter**.

```python
split_date = "2012-10-01"
train = serie.loc[:split_date]
test = serie.loc[split_date:]
print(train.shape, test.shape)
```

## Step 1 - Persistence baseline

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def persistence_forecast(y):
    return y.shift(1)

baseline_pred = persistence_forecast(test).dropna()
baseline_true = test.loc[baseline_pred.index]

mae_base = mean_absolute_error(baseline_true, baseline_pred)
rmse_base = mean_squared_error(baseline_true, baseline_pred, squared=False)
print(f"Persistence baseline - MAE: {mae_base:.2f}, RMSE: {rmse_base:.2f}")
```

- Keep these values as a **minimum reference**.
- Visualize predictions over the first 7 days to spot limits.

## Step 2 - Autoregressive AR(p) model

### 2.1 Choose the order

- Look at the PACF (from TP6.2) to propose 2-3 `p` values (e.g., 6, 12, 24).
- Option: use a manual GridSearch with AIC.

```python
from statsmodels.tsa.ar_model import AutoReg

orders = [6, 12, 24]
results = []
for p in orders:
    model = AutoReg(train, lags=p, old_names=False)
    model_fit = model.fit()
    pred = model_fit.predict(start=test.index[0], end=test.index[-1])
    rmse = mean_squared_error(test, pred, squared=False)
    results.append((p, model_fit.aic, rmse))

pd.DataFrame(results, columns=["p", "AIC", "RMSE"])
```

### 2.2 Analysis

- Choose the order that balances AIC and RMSE.
- Inspect `model_fit.params`: which lag values dominate?
- Analyze residuals:

```python
residuals = model_fit.resid
residuals.plot(title="AR model residuals")
plot_acf(residuals, lags=48)
```

**Questions**
- Do the residuals look random?  
- Should you adjust `p` or consider an MA component?

## Step 3 - ARIMA model

### 3.1 Choose (p, d, q)

- Set `d` based on TP6.2 (`d=1` if you have not already differenced the series).
- Use PACF for `p`, ACF for `q`. Test several combinations (e.g., `(2,1,2)`, `(5,1,0)`, `(3,1,3)`).

```python
from statsmodels.tsa.arima.model import ARIMA

def fit_arima(order):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    pred = model_fit.predict(start=test.index[0], end=test.index[-1])
    rmse = mean_squared_error(test, pred, squared=False)
    return model_fit, rmse

orders = [(2,1,2), (5,1,0), (3,1,3)]
report = []
for order in orders:
    model_fit, rmse = fit_arima(order)
    report.append((order, model_fit.aic, rmse))

pd.DataFrame(report, columns=["(p,d,q)", "AIC", "RMSE"])
```

### 3.2 Diagnostics and forecast

- Inspect `model_fit.summary()` and `model_fit.resid`.
- Check for residual autocorrelation (`plot_acf(model_fit.resid, lags=48)`).
- Produce a 7-day forecast:

```python
forecast = model_fit.get_forecast(steps=24*7)
forecast_ci = forecast.conf_int()

fig, ax = plt.subplots(figsize=(12,4))
test.iloc[:24*7].plot(ax=ax, label="Observed")
forecast.predicted_mean.plot(ax=ax, label="ARIMA forecast", color="red")
ax.fill_between(forecast_ci.index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1],
                color="red", alpha=0.2)
ax.legend()
```

**Questions**
- Does RMSE clearly improve the baseline?  
- Do confidence intervals cover the observations correctly?

## Step 4 - Comparison and limits

Gather metrics in a table:

| Model | MAE | RMSE | AIC (if applicable) |
| --- | --- | --- | --- |
| Persistence | ... | ... | - |
| AR(p=...) | ... | ... | ... |
| ARIMA(p,d,q) | ... | ... | ... |

**Analysis**
- Does ARIMA handle usage peaks better?  
- At what horizon does error increase quickly?
- What improvements would you consider (SARIMA, explanatory variables, hybrid models)?

## Deliverables

- Commented notebook with:
  - baseline, AR, ARIMA,
  - residual plots, forecast vs actual,
  - metrics table.
- Written synthesis (6-8 lines):
  1. Which model do you keep for short-term forecasting?
  2. Which risks remain (non-stationarity, anomalies)?
  3. Which additional data could improve performance?

Next step: put these results in perspective with supervised approaches in
[TP6.4](TP6_4_Modeles_Supervises_EN.md).
