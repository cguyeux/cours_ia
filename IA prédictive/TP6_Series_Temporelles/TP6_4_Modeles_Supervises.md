# TP6.4 — Approches supervisées et comparaison finale

[⬅️ Retour à l'index](TP6_Series_Temporelles_Index.md)

## Objectifs pédagogiques

- Transformer une série en dataset supervisé (features retardées, calendaires, moyennes mobiles).
- Mettre en place une validation glissante (`TimeSeriesSplit`) pour évaluer des modèles non sériels.
- Comparer forêts aléatoires, gradient boosting/XGBoost et ARIMA.
- Discuter des avantages/inconvénients des approches ML pour la prévision temporelle.

## Étape 1 — Construction des features

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

> Astuce : ajoutez les données météo (`temp`, `humidity`, …) pour tester l’apport d’informations exogènes.

## Étape 2 — Découpage temporel

- Conservez le même split que dans TP6.3 (`train jusqu'à fin septembre 2012`, `test = dernier trimestre`).
- Utilisez `TimeSeriesSplit` pour valider les hyperparamètres.

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

## Étape 3 — Modèles supervisés

### 3.1 Forêt aléatoire

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
print(f"Validation RF — MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
rmse_rf_test = mean_squared_error(y_test, pred_rf, squared=False)
```

### 3.2 XGBoost (ou Gradient Boosting)

Si `xgboost` n’est pas disponible, remplacez par `GradientBoostingRegressor`.

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
print(f"Validation XGB — MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")

xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
rmse_xgb_test = mean_squared_error(y_test, pred_xgb, squared=False)
```

### 3.3 Linearité de base (optionnel)

Tester une `Ridge` ou `Lasso` permet de voir si les lags suffisent sans modèle non linéaire.

## Étape 4 — Analyse des résultats

### 4.1 Tableau récapitulatif

| Modèle | MAE validation | RMSE validation | RMSE test |
| --- | --- | --- | --- |
| Persistance (TP6.3) | … | … | … |
| ARIMA (TP6.3) | … | … | … |
| RandomForest | `mae_rf` | `rmse_rf` | `rmse_rf_test` |
| XGBoost / GBoost | `mae_xgb` | `rmse_xgb` | `rmse_xgb_test` |

### 4.2 Visualisations

- Graphique comparant `y_test` vs `pred_rf` et `pred_xgb` sur 7 jours.
- Barres des features importances (RF ou XGBoost) :

```python
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).head(15).plot(kind="barh", figsize=(8,6))
```

- Analysez les features dominantes (lags, moyennes mobiles, heures).

### 4.3 Prévisions multi-horizon

- Implémentez un **rolling forecast** : à chaque pas, ajoutez la valeur prédite dans les features (pour un horizon > 1).
- Comparez la dérive des erreurs avec celle du modèle ARIMA.

## Étape 5 — Discussion critique

- Les modèles ML surpassent-ils ARIMA ? Sur quels aspects (pic, creux, lissage) ?
- Quel est le coût en termes d’interprétabilité, de maintenance (recalcul des features) ?
- Quelles données exogènes ajouteriez-vous (météo, calendrier scolaire, événements) ?

## Livrables finaux (parcours complet)

1. Notebook résumé (ou rapport) regroupant :
   - diagramme des transformations (TP6.1 → TP6.4),
   - tableau de métriques comparatif,
   - graphiques clés (décomposition, ACF, prévisions vs réalité, importances).
2. Recommandations opérationnelles :
   - taille de flotte quotidienne à prévoir,
   - procédures d’alerte (valeurs extrêmes / anomalies),
   - pistes d’amélioration (collecte de données exogènes, réentraînement).

Bravo, vous disposez désormais d’un pipeline complet de prévision temporelle, de l’exploration à la mise en production.
