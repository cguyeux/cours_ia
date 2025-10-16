# TP6.1 — Exploration et visualisation de la demande

[⬅️ Retour à l'index](TP6_Series_Temporelles_Index.md)

## Objectifs pédagogiques

- Mettre en place un pipeline de chargement propre pour une série horaire (`Bike_Sharing_Demand`).
- Vérifier la qualité des données (fréquence, valeurs manquantes, types).
- Identifier visuellement tendance globale, saisonnalités journalières/hebdomadaires et variabilité.
- Quantifier les dépendances temporelles via les autocorrélations et les lag plots.

## Mise en route

1. Créez un notebook `TP6_1_exploration.ipynb`.
2. Installez (si besoin) les dépendances : `pip install pandas matplotlib seaborn scikit-learn statsmodels`.
3. Téléchargez le dataset (extrait d’OpenML) :

```python
from sklearn.datasets import fetch_openml
import pandas as pd

raw = fetch_openml(name="Bike_Sharing_Demand", version=1, as_frame=True)
df = raw.frame
df["datetime"] = pd.to_datetime(df["datetime"])
serie = df.set_index("datetime")["count"].sort_index()
serie = serie.asfreq("H")  # impose la fréquence horaire
serie = serie.fillna(method="ffill")

serie.head()
serie.index.inferred_freq
```

> 💡 Conservez `df` : les colonnes météo (`temp`, `humidity`, etc.) pourront être utiles dans les sous-TP suivants.

## Étape 1 — Contrôle de la structure temporelle

- Vérifiez que l’index est un `DatetimeIndex`.
- Relevez la période couverte et le nombre d’observations (`serie.index.min()`, `serie.index.max()`, `serie.size`).
- Identifiez les éventuels trous après `asfreq("H")` (ex. `serie.isna().sum()` avant/après interpolation).

**À consigner**
- Comment gérez-vous les rares valeurs manquantes ? Quelle hypothèse implique la propagation `ffill` ?
- Quelles colonnes supplémentaires de `df` pourraient enrichir l’analyse plus tard ?

## Étape 2 — Exploration rapide

1. Affichez `serie.head(24)` et `serie.tail(24)` pour vérifier la cohérence horaire.
2. Mesurez les statistiques descriptives globales :

```python
serie.describe()
serie.resample("D").sum().describe()
```

3. Analysez la distribution avec un histogramme et un boxplot :

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
serie.plot(kind="hist", bins=40, alpha=0.7)
plt.title("Histogramme des locations horaires")
plt.show()

serie.to_frame("count").boxplot(figsize=(4,6))
```

**Questions guidées**
- Quelle est la médiane des locations horaires ? À quoi correspond-elle opérationnellement ?
- L’histogramme présente-t-il une queue longue ? Comment l’expliquer (météo, événements ponctuels…) ?

## Étape 3 — Visualisations essentielles

### 3.1 Vue globale

```python
plt.figure(figsize=(16,4))
serie.plot(title="Série horaire des locations (2011-2012)")
plt.xlabel("Date")
plt.ylabel("Nombre de locations")
plt.show()
```

- Interprétez la tendance générale (croissance ? stagnation ?).
- Repérez à l’œil les pics récurrents (week-ends, événements, météo…).

### 3.2 Variations intra-annuelles

```python
years = serie.groupby(pd.Grouper(freq="A"))
df_years = pd.DataFrame({str(name.year): group.values for name, group in years})
df_years.plot(figsize=(14,10), subplots=True, sharex=False, sharey=True, legend=False)
```

- Comparez 2011 vs 2012 : la demande augmente-t-elle ? les pics sont-ils plus fréquents ?

### 3.3 Cycle hebdomadaire et quotidien

```python
serie.resample("D").sum().plot(figsize=(12,4), title="Locations quotidiennes")
plt.show()

serie.groupby(serie.index.hour).mean().plot(kind="bar", figsize=(10,4))
plt.title("Profil moyen par heure")
plt.show()

serie.groupby(serie.index.day_name()).mean().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
).plot(kind="bar", figsize=(10,4))
plt.title("Profil moyen par jour de la semaine")
plt.show()
```

- Quelles plages horaires sont critiques pour la logistique ?  
- Les week-ends suivent-ils la même tendance que les jours ouvrés ?

## Étape 4 — Autocorrélation et dépendances temporelles

### 4.1 Lag plots

```python
pd.plotting.lag_plot(serie, lag=1, s=5, alpha=0.3)
pd.plotting.lag_plot(serie, lag=24, s=5, alpha=0.3)
```

- Comparez la structure des nuages de points pour `lag=1` et `lag=24`.  
- Que signifient les diagonales visibles ?

### 4.2 Fonctions d’autocorrélation

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14,4))
plot_acf(serie, lags=48, ax=axes[0])
plot_pacf(serie, lags=48, ax=axes[1], method="ywm")
plt.show()
```

- Identifiez les lags significatifs (24, 168, …).  
- Notez-les : ils guideront le choix de `p` et `q` dans le sous-TP 6.3.

## Synthèse & livrables

- Tableau de bord minimal (dans le notebook) récapitulant :
  - statistiques globales (`mean`, `median`, `max`, `std`),
  - graphique global,
  - distribution horaire/hebdomadaire,
  - ACF/PACF.
- Paragraphe de synthèse (5 lignes) répondant aux questions :
  1. Quels motifs saisonniers sont observés ?
  2. Quel niveau de demande doit-on anticiper en heures de pointe ?
  3. Quels premiers soupçons sur la stationnarité de la série ?

Prochaine étape : passer à [TP6.2](TP6_2_Decomposition_Stationnarite.md) pour séparer tendance et saisonnalité et préparer les
modèles AR/ARIMA.
