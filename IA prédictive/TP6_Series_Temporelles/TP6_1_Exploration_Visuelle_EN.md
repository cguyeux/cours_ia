# TP6.1 - Demand exploration and visualization

[Back to index](TP6_Series_Temporelles_Index_EN.md)

## Learning objectives

- Build a clean loading pipeline for an hourly series (`Bike_Sharing_Demand`).
- Check data quality (frequency, missing values, types).
- Visually identify overall trend, daily/weekly seasonality, and variability.
- Quantify temporal dependencies via autocorrelations and lag plots.

## Getting started

1. Create a notebook `TP6_1_exploration.ipynb`.
2. Install dependencies (if needed): `pip install pandas matplotlib seaborn scikit-learn statsmodels`.
3. Download the provided dataset (train.csv):

```python
from sklearn.datasets import fetch_openml
import pandas as pd

df = pd.read_csv("train.csv", parse_dates=["datetime"])

serie = df.set_index("datetime")["count"].sort_index()

print(f"Missing values: {serie.isna().sum()}")
serie = serie.asfreq("h")  # enforce hourly frequency
serie = serie.ffill()
print(f"Missing values: {serie.isna().sum()}")

serie.index.inferred_freq

serie.head()
```

Keep `df`: weather columns (`temp`, `humidity`, etc.) can be useful in later sub-labs.

## Step 1 - Check the time structure

- Verify the index is a `DatetimeIndex`.
- Record the time span and number of observations (`serie.index.min()`, `serie.index.max()`, `serie.size`).
- Identify any gaps after `asfreq("H")` (e.g., `serie.isna().sum()` before/after interpolation).

**To record**
- How do you handle rare missing values? What assumption does `ffill` imply?
- Which additional columns from `df` could enrich the analysis later?

## Step 2 - Quick exploration

1. Display `serie.head(24)` and `serie.tail(24)` to check hourly consistency.
2. Measure global descriptive statistics:

```python
serie.describe()
serie.resample("D").sum().describe()
```

3. Analyze the distribution with a histogram and a boxplot:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
serie.plot(kind="hist", bins=40, alpha=0.7)
plt.title("Histogram of hourly rentals")
plt.show()

serie.to_frame("count").boxplot(figsize=(4,6))
```

**Guided questions**
- What is the median of hourly rentals? What does it correspond to operationally?
- Does the histogram show a long tail? How can you explain it (weather, events)?

## Step 3 - Essential visualizations

### 3.1 Global view

```python
plt.figure(figsize=(16,4))
serie.plot(title="Hourly rentals (2011-2012)")
plt.xlabel("Date")
plt.ylabel("Number of rentals")
plt.show()
```

- Interpret the overall trend (growth? stagnation?).
- Spot recurring peaks (weekends, events, weather).

### 3.2 Intra-year variations

```python
years = serie.groupby(pd.Grouper(freq="A"))
df_years = pd.DataFrame({str(name.year): group.values for name, group in years})
df_years.plot(figsize=(14,10), subplots=True, sharex=False, sharey=True, legend=False)
```

- Compare 2011 vs 2012: is demand increasing? are peaks more frequent?

### 3.3 Weekly and daily cycle

```python
serie.resample("D").sum().plot(figsize=(12,4), title="Daily rentals")
plt.show()

serie.groupby(serie.index.hour).mean().plot(kind="bar", figsize=(10,4))
plt.title("Average profile by hour")
plt.show()

serie.groupby(serie.index.day_name()).mean().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
).plot(kind="bar", figsize=(10,4))
plt.title("Average profile by day of week")
plt.show()
```

- Which time ranges are critical for operations?  
- Do weekends follow the same trend as weekdays?

## Step 4 - Autocorrelation and temporal dependencies

### 4.1 Lag plots

```python
pd.plotting.lag_plot(serie, lag=1, s=5, alpha=0.3)
pd.plotting.lag_plot(serie, lag=24, s=5, alpha=0.3)
```

- Compare the point-cloud structure for `lag=1` and `lag=24`.  
- What do the visible diagonals mean?

### 4.2 Autocorrelation functions

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14,4))
plot_acf(serie, lags=48, ax=axes[0])
plot_pacf(serie, lags=48, ax=axes[1], method="ywm")
plt.show()
```

- Identify significant lags (24, 168, ...).  
- Note them: they will guide the choice of `p` and `q` in sub-lab 6.3.

## Synthesis and deliverables

- A minimal dashboard (in the notebook) summarizing:
  - global statistics (`mean`, `median`, `max`, `std`),
  - global plot,
  - hourly/weekly distribution,
  - ACF/PACF.
- A short synthesis paragraph (5 lines) answering:
  1. Which seasonal patterns are observed?
  2. What demand level should be anticipated at peak hours?
  3. What first suspicion do you have about stationarity?

Next step: move to [TP6.2](TP6_2_Decomposition_Stationnarite_EN.md) to separate trend and seasonality and prepare AR/ARIMA models.
