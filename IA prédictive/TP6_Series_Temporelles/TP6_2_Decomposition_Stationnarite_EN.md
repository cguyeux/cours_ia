# TP6.2 - Trend, seasonality, and stationarity

[Back to index](TP6_Series_Temporelles_Index_EN.md)

## Learning objectives

- Identify the components of a series (level, trend, seasonality, noise).
- Choose between additive or multiplicative decomposition.
- Make a series stationary with simple and seasonal differencing.
- Document the impact of these transformations on analysis and modeling.

Starting point: reuse the `serie` created in [TP6.1](TP6_1_Exploration_Visuelle_EN.md). If you work in a new notebook, reload the dataset and keep the same preprocessing steps.

## Step 1 - Understand the components

In an additive model:  
$y_t = \text{level} + \text{trend}_t + \text{seasonality}_t + \text{residual}_t$  
In a multiplicative model:  
$y_t = \text{level} \times \text{trend}_t \times \text{seasonality}_t \times \text{residual}_t$.

**Reflection questions**
- Does peak amplitude increase with level (=> multiplicative) or remain stable (=> additive)?
- Which time scales seem relevant (daily, weekly)?

## Step 2 - Seasonal decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

result_add = seasonal_decompose(serie, model="additive", period=24)
result_add.plot()
plt.suptitle("Additive decomposition - 24h period", y=1.02)
plt.show()

result_week = seasonal_decompose(serie, model="multiplicative", period=24*7)
result_week.plot()
plt.suptitle("Multiplicative decomposition - 7-day period", y=1.02)
plt.show()
```

- Comment on the trend (does it increase during 2012?).  
- Are seasonal patterns close to a sinusoid? Observations about weekends?
- Does the residual look like noise (centered values)?

**To produce**: a short paragraph (4-5 lines) describing what these graphs say for the business.

## Step 3 - Stationarity tests

Stationarity (constant mean and variance) is crucial for AR/ARIMA models.

```python
from statsmodels.tsa.stattools import adfuller

def adf_report(series, name):
    stat, pvalue, *_ = adfuller(series.dropna())
    print(f"ADF test on {name} - Statistic: {stat:.3f}, p-value: {pvalue:.3f}")

adf_report(serie, "raw series")
adf_report(result_add.resid, "residual (additive model)")
```

- Interpret the p-value: < 0.05 => likely stationary.  
- Is the residual stationary? Why does it matter?

## Step 4 - Differencing

### 4.1 Simple differencing (trend)

```python
diff1 = serie.diff().dropna()
plt.figure(figsize=(12,3))
diff1.plot(title="Simple differencing (order 1)")
plt.show()
adf_report(diff1, "first-order difference")
```

- How does the variance evolve?  
- Does the ADF test conclude stationarity?

### 4.2 Seasonal differencing (24 hours)

```python
diff_season = serie.diff(24).dropna()
plt.figure(figsize=(12,3))
diff_season.plot(title="Seasonal differencing (24h period)")
plt.show()
adf_report(diff_season, "24h difference")
```

- Compare `diff1` and `diff_season`. Which one makes daily patterns less visible?
- Test a combination: `serie.diff(24).diff().dropna()`.

### 4.3 Impact on autocorrelation

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14,4))
plot_acf(diff1, lags=48, ax=axes[0])
plot_pacf(diff1, lags=48, ax=axes[1], method="ywm")
plt.suptitle("ACF/PACF after simple differencing", y=1.05)
plt.show()
```

- Did significant lags change?  
- Which AR (`p`) and MA (`q`) orders would you consider now?

## Step 5 - Reconstruction and interpretation

- Compute `serie_hat = (diff1.cumsum() + serie.iloc[0])` to visualize the return to the original scale.
- Consider applying `np.log1p(serie)` before differencing if variance remains level-dependent.

**Deliverables**
- Decomposition plots (24h and 7-day periods).
- Summary table of ADF tests (raw series, residual, simple diff, seasonal diff).
- Short synthesis (5 lines): which transformations will you keep for the next step, and why?

Next step: apply these findings in [TP6.3 - AR and ARIMA models](TP6_3_Modeles_AR_ARIMA_EN.md).
