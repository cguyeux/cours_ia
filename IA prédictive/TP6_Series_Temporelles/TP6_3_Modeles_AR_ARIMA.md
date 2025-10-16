# TP6.3 — Modèles AR et ARIMA

[⬅️ Retour à l'index](TP6_Series_Temporelles_Index.md)

## Objectifs pédagogiques

- Mettre en place une **baseline de persistance** pour juger les gains des modèles avancés.
- Entraîner et interpréter un **modèle auto-régressif AR(p)**.
- Paramétrer un **ARIMA(p,d,q)** cohérent avec les diagnostics de stationnarité.
- Évaluer les modèles sur des prévisions de court terme et analyser les résidus.

## Préparation

- Utilisez la série transformée lors du TP6.2 (différenciation si nécessaire).
- Définissez un découpage temporel : par exemple, **train = données jusqu'au 30 septembre 2012**, **test = dernier trimestre**.

```python
split_date = "2012-10-01"
train = serie.loc[:split_date]
test = serie.loc[split_date:]
print(train.shape, test.shape)
```

## Étape 1 — Baseline de persistance

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
print(f"Baseline persistance — MAE: {mae_base:.2f}, RMSE: {rmse_base:.2f}")
```

- Gardez ces valeurs comme **référence minimale**.
- Visualisez les prédictions sur les 7 premiers jours pour repérer les limites.

## Étape 2 — Modèle auto-régressif AR(p)

### 2.1 Choisir l’ordre

- Observez la PACF (depuis TP6.2) pour proposer 2-3 valeurs de `p` (ex. 6, 12, 24).
- Option : utiliser un `GridSearch` manuel avec l’AIC.

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

### 2.2 Analyse

- Choisissez l’ordre qui équilibre bien AIC et RMSE.
- Étudiez `model_fit.params`: quelles valeurs de lags sont majeures ?
- Analysez les résidus :

```python
residuals = model_fit.resid
residuals.plot(title="Résidus du modèle AR")
plot_acf(residuals, lags=48)
```

**Questions**
- Les résidus semblent-ils aléatoires ?  
- Faut-il ajuster `p` ou envisager une composante MA ?

## Étape 3 — Modèle ARIMA

### 3.1 Choisir (p, d, q)

- Définissez `d` en fonction du TP6.2 (`d=1` si vous n'avez pas déjà différencié la série).
- Utilisez la PACF pour `p`, l'ACF pour `q`. Testez plusieurs combinaisons (ex. `(2,1,2)`, `(5,1,0)`, `(3,1,3)`).

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

### 3.2 Diagnostic & prévision

- Inspectez `model_fit.summary()` et `model_fit.resid`.
- Vérifiez l’absence d’autocorrélation des résidus (`plot_acf(model_fit.resid, lags=48)`).
- Réalisez une prévision à horizon 7 jours :

```python
forecast = model_fit.get_forecast(steps=24*7)
forecast_ci = forecast.conf_int()

fig, ax = plt.subplots(figsize=(12,4))
test.iloc[:24*7].plot(ax=ax, label="Observé")
forecast.predicted_mean.plot(ax=ax, label="Prévision ARIMA", color="red")
ax.fill_between(forecast_ci.index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1],
                color="red", alpha=0.2)
ax.legend()
```

**Questions**
- Le RMSE améliore-t-il nettement la baseline ?  
- Les intervalles de confiance couvrent-ils correctement les observations ?

## Étape 4 — Comparaison et limites

Rassemblez les métriques dans un tableau :

| Modèle | MAE | RMSE | AIC (si applicable) |
| --- | --- | --- | --- |
| Persistance | … | … | — |
| AR(p=…) | … | … | … |
| ARIMA(p,d,q) | … | … | … |

**Analyse**
- Le modèle ARIMA se comporte-t-il mieux sur les pics d’utilisation ?  
- À quel horizon l’erreur augmente-t-elle rapidement ?
- Quelles améliorations envisager (SARIMA, variables explicatives, hybridation) ?

## Livrables

- Notebook commenté avec :
  - baseline, AR, ARIMA,
  - graphiques de résidus, prévisions vs réalité,
  - tableau de métriques.
- Synthèse écrite (6-8 lignes) :
  1. Quel modèle retenez-vous pour la prévision à court terme ?
  2. Quels risques subsistent (non-stationnarité, anomalies) ?
  3. Quelles données additionnelles pourraient améliorer la performance ?

Prochaine étape : mettre ces résultats en perspective avec des approches supervisées dans
[TP6.4](TP6_4_Modeles_Supervises.md).
