# TP6 - Time Series: Guided Path

[Back to contents](../../LISEZMOI.md)

## Why this path?

Time series data are everywhere (IoT, mobility, transport, energy). This lab offers a progressive thread: start from a raw series, understand it, stabilize it, then compare several families of forecasting models. The work is split into four independent sub-labs (about 1.5 to 2 hours each) to fit your availability.

## Business scenario

You assist the Washington DC bike-share operator. Your mission: **anticipate hourly demand** to better regulate the fleet. The reference dataset is `Bike_Sharing_Demand` (OpenML #42712), containing actual rentals between 2011 and 2012.

## Guiding thread

1. **Explore** the series to understand its patterns (daily peaks, weekends, seasons).
2. **Separate** trend, seasonality, and noise to stabilize the signal.
3. **Test** classic time-series models (persistence, AR, ARIMA) and understand their limits.
4. **Compare** with supervised models using lagged features and exogenous variables.

Each sub-lab ends with a mini-synthesis that feeds a final report.

## Recommended organization

- **Indicative duration**: 6 to 8 hours total.
- **Environment**: Python >= 3.9, `pandas`, `matplotlib`/`seaborn`, `statsmodels`, `scikit-learn`, `xgboost` (optional).
- **Global deliverable**: a notebook or report consolidating your results plus operational recommendations.

Tip: create a `notebooks/TP6` folder and one notebook per sub-lab so you can easily return to your analyses.

## Four sub-labs

1. [**TP6.1 - Exploration and visualization of demand**](TP6_1_Exploration_Visuelle_EN.md)  
   Load the dataset, perform quality checks, produce global and cyclic visualizations, and study autocorrelations.

2. [**TP6.2 - Trend, seasonality, and stationarity**](TP6_2_Decomposition_Stationnarite_EN.md)  
   Additive/multiplicative decomposition, simple and seasonal differencing, ADF tests.

3. [**TP6.3 - AR and ARIMA models**](TP6_3_Modeles_AR_ARIMA_EN.md)  
   Persistence baseline, AR order selection, ARIMA training, residual analysis, short-term forecasts.

4. [**TP6.4 - Supervised approaches and final comparison**](TP6_4_Modeles_Supervises_EN.md)  
   Lagged features, rolling validation, Random Forest / (X)GBoost, comparison with ARIMA.

## Suggested deliverables

- Comparative MAE/RMSE table (baseline vs AR vs ARIMA vs supervised model).
- Key plots: decomposition, ACF/PACF, forecast vs actual, feature importances.
- Recommendations for the operator (fleet management, need for exogenous data, retraining frequency).

## Useful resources

- [`statsmodels` ARIMA documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html).
- [`scikit-learn` TimeSeriesSplit guide](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).
- For further work: `pmdarima.auto_arima`, SARIMAX, Prophet, LSTM/TCN networks.

Good luck! Do not hesitate to note your assumptions and limits throughout the lab: they will feed your final synthesis.
