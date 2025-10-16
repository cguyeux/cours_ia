# TP6.2 — Tendance, saisonnalité et stationnarité

[⬅️ Retour à l'index](TP6_Series_Temporelles_Index.md)

## Objectifs pédagogiques

- Identifier les composantes d’une série (niveau, tendance, saisonnalité, bruit).
- Choisir entre décomposition additive ou multiplicative.
- Rendre une série stationnaire grâce aux différenciations simple et saisonnière.
- Documenter l’impact de ces transformations sur l’analyse et la modélisation.

> Point de départ : réutilisez la série `serie` créée dans [TP6.1](TP6_1_Exploration_Visuelle.md). Si vous travaillez dans un nouveau
> notebook, rechargez le dataset et conservez les mêmes pré-traitements.

## Étape 1 — Comprendre les composantes

Dans un modèle additif :  
$y_t = \text{niveau} + \text{tendance}_t + \text{saisonnalité}_t + \text{résidu}_t$  
Dans un modèle multiplicatif :  
$y_t = \text{niveau} \times \text{tendance}_t \times \text{saisonnalité}_t \times \text{résidu}_t$.

**Questions de réflexion**
- L’amplitude des pics augmente-t-elle avec le niveau (=> multiplicatif) ou reste-t-elle stable (=> additif) ?
- Quelles échelles temporelles semblent pertinentes (journalier, hebdomadaire) ?

## Étape 2 — Décomposition saisonnière

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

result_add = seasonal_decompose(serie, model="additive", period=24)
result_add.plot()
plt.suptitle("Décomposition additive - période 24h", y=1.02)
plt.show()

result_week = seasonal_decompose(serie, model="multiplicative", period=24*7)
result_week.plot()
plt.suptitle("Décomposition multiplicative - période 7 jours", y=1.02)
plt.show()
```

- Commentez la tendance (croît-elle sur l’année 2012 ?).  
- Les motifs saisonniers sont-ils proches d’une sinusoïde ? Observations sur les week-ends ?
- Le résidu ressemble-t-il à un bruit (valeurs centrées) ?

**À produire** : un court paragraphe (4-5 lignes) décrivant ce que racontent ces graphes au métier.

## Étape 3 — Tests de stationnarité

La stationnarité (moyenne et variance constantes) est cruciale pour les modèles AR/ARIMA.

```python
from statsmodels.tsa.stattools import adfuller

def adf_report(series, name):
    stat, pvalue, *_ = adfuller(series.dropna())
    print(f"Test ADF sur {name} — Statistique: {stat:.3f}, p-value: {pvalue:.3f}")

adf_report(serie, "série brute")
adf_report(result_add.resid, "résidu (modèle additif)")
```

- Interprétez la p-value : < 0,05 => stationnarité probable.  
- Le résidu est-il stationnaire ? Pourquoi est-ce important ?

## Étape 4 — Différenciation

### 4.1 Différenciation simple (tendance)

```python
diff1 = serie.diff().dropna()
plt.figure(figsize=(12,3))
diff1.plot(title="Différenciation simple (ordre 1)")
plt.show()
adf_report(diff1, "différence ordre 1")
```

- Comment la variance évolue-t-elle ?  
- Le test ADF conclut-il à la stationnarité ?

### 4.2 Différenciation saisonnière (24 heures)

```python
diff_season = serie.diff(24).dropna()
plt.figure(figsize=(12,3))
diff_season.plot(title="Différenciation saisonnière (période 24h)")
plt.show()
adf_report(diff_season, "différence 24h")
```

- Comparez `diff1` et `diff_season`. Laquelle rend les motifs journaliers moins visibles ?
- Testez une combinaison : `serie.diff(24).diff().dropna()`.

### 4.3 Impact sur l'autocorrélation

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14,4))
plot_acf(diff1, lags=48, ax=axes[0])
plot_pacf(diff1, lags=48, ax=axes[1], method="ywm")
plt.suptitle("ACF/PACF après différenciation simple", y=1.05)
plt.show()
```

- Les lags significatifs ont-ils changé ?  
- Quels ordres AR (`p`) et MA (`q`) envisagez-vous désormais ?

## Étape 5 — Reconstitution & interprétation

- Calculez `serie_hat = (diff1.cumsum() + serie.iloc[0])` pour visualiser le retour à l’échelle originale.
- Envisagez d’appliquer une transformation `np.log1p(serie)` avant différenciation si la variance reste dépendante du niveau.

**Livrables**
- Graphiques de décomposition (période 24h et 7j).
- Tableau récapitulatif des tests ADF (série brute, résidu, diff simple, diff saisonnière).
- Commentaire synthétique (5 lignes) : quelles transformations retiendrez-vous pour la suite et pourquoi ?

Prochaine étape : appliquer ces enseignements dans [TP6.3 — Modèles AR et ARIMA](TP6_3_Modeles_AR_ARIMA.md).
