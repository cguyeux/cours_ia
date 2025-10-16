# TP : Séries temporelles et prévision

[⬅️ Retour au sommaire](../../LISEZMOI.md)

## Objectifs pédagogiques

- Comprendre les notions fondamentales associées aux séries temporelles (tendance, saisonnalité, bruit, stationnarité).
- Mettre en place un pipeline d'analyse exploratoire spécifique aux données indexées dans le temps.
- Construire des jeux d'entraînement / validation adaptés et créer des variables retardées (`lag features`).
- Comparer des approches de prévision : persistance, régression supervisée avec fenêtres glissantes, modèles ARIMA.
- Évaluer la qualité des prévisions à l'aide de métriques adaptées et interpréter les résultats.

## Prérequis

- Python, `pandas`, `numpy`, `matplotlib` ou `seaborn`.
- Notions de régression (voir [TP5 – Régression](../TP5_Regression/TP5_Regression_Validation.md)).
- Installation de `statsmodels` (disponible dans `scikit-learn` Anaconda) pour l'accès aux jeux de données.

## Contexte

Les séries temporelles sont omniprésentes : mesures de capteurs IoT, consommation énergétique, trafic d'un site web, température,
ventes quotidiennes, etc. Contrairement aux jeux de données « tabulaires » classiques, l'ordre chronologique porte de l'information.
Ignorer cette dimension temporelle peut conduire à des modèles biaisés ou à des évaluations trompeuses.

Dans ce TP, vous travaillez sur une série de concentration en $\mathrm{CO_2}$ atmosphérique (jeu de données `co2` fourni par
`statsmodels`). Les mesures hebdomadaires ont été collectées à Mauna Loa (Hawaï) pendant plusieurs décennies.

## Visualisation des données pour ce TP

Les captures ci-dessous synthétisent les étapes indispensables avant toute modélisation. Elles illustrent l'exploration d'une série
horaire (nombre d'interventions des pompiers) ; nous allons reproduire les mêmes gestes sur un dataset disponible via `sklearn`.

> Jeu de données support : `Bike_Sharing_Demand` (OpenML 42712). Il contient les locations de vélos partagés à Washington DC et une
> colonne `datetime` horaire.

```python
from sklearn.datasets import fetch_openml
import pandas as pd

raw = fetch_openml(name="Bike_Sharing_Demand", version=1, as_frame=True)
df = raw.frame
df["datetime"] = pd.to_datetime(df["datetime"])
serie = df.set_index("datetime")["count"].asfreq("H")
serie = serie.fillna(method="ffill")  # comble les rares trous horaires
```

### Lecture du fichier et contrôle du type temporel

- La commande `pd.read_csv(..., parse_dates=True, index_col=0, squeeze=True)` transforme immédiatement la première colonne en index
  temporel. Cette étape garantit que `pandas` comprenne la notion de **chronologie**.
- **Exercice.** Vérifiez que l'index de `serie` est bien de type `DatetimeIndex`. Que se passe-t-il si vous omettez `asfreq("H")` ?
  Illustrez la différence en affichant `serie.index.inferred_freq`.

### Découverte rapide

- `serie.head()` et `serie.tail()` permettent de vérifier l'ordre des dates et l'unité de temps (heures vs jours). La méthode
  `size` donne le volume d'observations.
- **Exercice.** Affichez `serie.loc["2011-01-03"]`. Comment interpréter les 24 lignes retournées ?

### Navigation par index temporel

- `serie["2012-08-15 09:00:00"]` retourne la valeur exacte à une heure donnée ; `serie["2012-08"]` filtre sur tout un mois ;
  `serie["2011-06-01 07:00":"2011-06-01 12:00"]` extrait un intervalle.
- **Exercice.** Mesurez l'activité moyenne par créneau entre 7h et 9h sur la semaine du 4 juillet 2011. Que concluez-vous sur la
  fréquentation matinale ?

### Statistiques descriptives

- `serie.describe()` fournit volume, moyenne, quantiles et maximum : c'est une photographie de la distribution.
- Exemple métier : la médiane (ici ≈ 145 locations/h) aide l'opérateur à dimensionner le stock de vélos disponible.
- **Exercice.** Comparez `serie.describe()` sur les mois d'hiver (`serie["2011-12"]`) et d'été (`serie["2011-07"]`). Que remarquez-vous ?

### Ingénierie de variables explicatives

- On extrait des composantes temporelles (`index.day`, `index.hour`) pour créer des variables métiers (jour du mois, heure). Cette
  étape prépare l'entraînement d'un modèle supervisé classique.
- Les décalages (`serie.shift(1)`, `serie.shift(12)`) donnent accès aux valeurs passées ; ils matérialisent la dépendance temporelle.
- Les moyennes mobiles (`serie.shift(1).rolling(window=3).mean()`) lissent la série et capturent la tendance court terme.
- **Exercice.** Créez un DataFrame `features` contenant les colonnes `count_t`, `count_t-1`, `count_t-24`, `rolling_mean_7` (moyenne
  mobile journalière) et `is_weekend`. Identifiez les trois variables les plus corrélées avec `count_t`.

### Visualisations globales

- `serie.plot()` met en évidence tendance et pics : indispensable pour détecter dérives ou ruptures. Sur le partage de vélos, on
  observe une croissance nette entre 2011 et 2012 ainsi que des pointes saisonnières.
- Une agrégation par année (`serie.groupby(pd.Grouper(freq="A"))`) puis un tracé par sous-graphiques permet de comparer les profils
  annuels : utile pour repérer un changement d'usage après une extension du service.
- **Exercice.** Reproduisez la figure multi-année : quelles années présentent les plus forts pics ? Proposez une hypothèse métier.

### Distribution des valeurs

- L'histogramme (`serie.hist(bins=30)`) montre la fréquence des heures calmes vs saturées ; c'est primordial pour dimensionner les
  équipes ou la maintenance.
- Le graphique de densité (`serie.plot(kind="kde")`) met en lumière la présence d'une queue longue (heures exceptionnellement
  chargées).
- Les boxplots par année/mois (`serie.groupby(...).boxplot()`) comparent l'étendue et les outliers entre saisons.
- **Exercice.** Construisez un boxplot par mois sur l'année 2012. Quels mois produisent le plus d'outliers ? Comment expliquer ce
  phénomène (météo, événements, vacances...) ?

> Ces visualisations constituent votre **check-list** avant de passer à la modélisation : elles orientent le choix des features, la
> granularité des modèles et les métriques à surveiller.

### Autocorrélation et corrélations retardées

- **Définition.** L'autocorrélation au lag $k$ mesure la corrélation linéaire entre $y_t$ et $y_{t-k}$. Elle quantifie la mémoire de la
  série : une valeur proche de 1 signifie que la série « se ressemble » d'une période à l'autre.
- **Pourquoi la mesurer ?** Pour identifier des dépendances temporelles (ex. effet d'inertie, saisons hebdomadaires) et guider le
  choix d'un horizon de prédiction ou d'un modèle (ARIMA, régression avec lags).

```python
serie.autocorr(lag=1)      # corrélation entre t et t-1
serie.autocorr(lag=24)     # corrélation sur un décalage journalier
```

- **Lag plots.** `pd.plotting.lag_plot(serie, lag=k)` affiche un nuage de points $(y_{t-k}, y_t)$. Une diagonale marquée révèle une
  forte autocorrélation. La fonction `pd.plotting.lag_plot` peut être utilisée en sous-graphiques pour plusieurs lags.
- **Fonctions ACF/PACF.** Utilisez `statsmodels.graphics.tsaplots.plot_acf` et `plot_pacf` pour visualiser les autocorrélations
  sur plusieurs dizaines de lags avec des intervalles de confiance.

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sm.graphics.tsa.plot_acf(serie, lags=48, ax=axes[0])
sm.graphics.tsa.plot_pacf(serie, lags=48, ax=axes[1], method="ywm")
```

- **Exemple métier.** Sur les locations de vélos, un pic d'autocorrélation à $k=24$ signale un comportement similaire d'un jour sur
  l'autre ; un pic à $k=24 \times 7$ met en évidence la routine hebdomadaire (week-ends plus calmes).

- **Exercice 1.** Calculez et interprétez `serie.autocorr(lag=1)`, `lag=24` et `lag=168`. Quelle lecture en faites-vous pour le
  service de vélos ? Lesquels de ces lags intégreriez-vous dans vos features ?
- **Exercice 2.** Tracez un lag plot pour `lag=1` et `lag=24`. Comparez la dispersion des points et expliquez la différence.
- **Exercice 3.** À l'aide de `plot_acf`, identifiez le premier lag dont la valeur retombe dans l'intervalle de confiance (zone
  gris clair). Que signifie ce point pour le paramètre $q$ d'un modèle ARIMA ?

### Tendance et saisonnalité

- **Composantes d'une série.** Toute série peut être vue comme une combinaison d'un niveau moyen, d'une **tendance** (croissance ou
  décroissance à long terme), d'une **saisonnalité** (motif périodique) et d'un **résidu** (bruit). On distingue les modèles
  additifs ($y_t = \text{niveau} + \text{tendance} + \text{saisonnalité} + \text{bruit}$) et multiplicatifs
  ($y_t = \text{niveau} \times \text{tendance} \times \text{saisonnalité} \times \text{bruit}$). La nature du modèle dépend de la
  manière dont l'amplitude saisonnière varie : constante (additif) ou proportionnelle au niveau (multiplicatif).

- **Décomposition automatique.** `statsmodels.tsa.seasonal_decompose` sépare la série selon une période donnée :

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(serie, model="additive", period=24)
result.plot()
plt.show()
```

  - **Trend** : évolution lissée (ex. hausse progressive des locations de vélos avec l'expansion du service).
  - **Seasonal** : motif récurrent (ex. pics matin/soir chaque jour).
  - **Resid** : ce qui reste à expliquer (aléas météo, événements ponctuels).

- **Exemple métier.** Pour un gestionnaire de flotte, la tendance aide à planifier les achats de vélos sur l'année ; la saisonnalité
  horaire indique les horaires de renfort d'équipes ; le résidu alerte sur des anomalies (pannes massives, grèves...).

- **Choisir la période.** Pour la série horaire des vélos, testez `period=24` (cycle journalier) puis `period=24*7` (cycle
  hebdomadaire). Comparez les graphiques obtenus.

- **Exercice 1.** Exécutez la décomposition additive avec `period=24` et interprétez chaque composante. Quels éléments confirment vos
  observations issues des histogrammes et de l'autocorrélation ?
- **Exercice 2.** Changez `model="multiplicative"` et comparez la courbe saisonnière. Dans quel cas ce modèle serait-il plus adapté
  (indice : lorsque l'amplitude des pics augmente avec le niveau moyen) ?

- **Suppression de tendance (differencing).** Pour travailler sur une série stationnaire, on peut différencier :

```python
diff = serie.diff().dropna()
```

  Cela réduit les dérives lentes et facilite l'entraînement de modèles ARIMA. Visualisez `diff.plot()` pour vérifier que la tendance
  a été atténuée.

- **Ajustement saisonnier.** En soustrayant la composante saisonnière estimée (`serie - result.seasonal`) ou en utilisant
  `diff = serie.diff(periods=24)`, on obtient une série plus stable. Utile pour isoler les effets météo ou marketing.

- **Exercice 3.** Calculez `serie.diff(periods=24)` et comparez la variance avant/après. Quelle conclusion tirez-vous pour le choix
  de la période de différenciation dans un modèle SARIMA ?
- **Exercice 4.** Construisez un modèle simple de saisonnalité en ajustant un polynôme sur les heures de la journée (ex.
  `np.polyfit` sur `hour -> count`). Superposez ce polynôme à la courbe d'un jour typique : que met-il en évidence ?
- **Exercice 5.** Retirez la saisonnalité estimée et affichez les résidus. Quels créneaux horaires présentent encore des
  irrégularités fortes ? Formulez une hypothèse métier (météo, événements, vacances scolaires...).

## Partie 1 – Exploration temporelle

### 1.1 Chargement et nettoyage (notebook)

- Importez la série via :

```python
import statsmodels.api as sm

co2 = sm.datasets.co2.load_pandas().data
co2 = co2.rename(columns={"co2": "ppm"})
co2 = co2.asfreq("W")  # fréquence hebdomadaire
co2 = co2.fillna(method="ffill")  # valeurs manquantes
```

- **Question 1.** Pourquoi est-il important de spécifier la fréquence (`asfreq`) pour une série temporelle ? Quelles conséquences si
on ne le fait pas ?

### 1.2 Visualisations essentielles

1. Tracez l'évolution globale (`plot`) et commentez les composantes tendance / saisonnalité / résidus que vous observez.
2. Affichez la série par année (ex. avec `pivot_table` ou `seaborn.lineplot`). Que révèle cette visualisation ?
3. Calculez l'autocorrélation simple (`Series.autocorr(lag=1)`), puis dessinez les fonctions d'autocorrélation (ACF) et
d'autocorrélation partielle (PACF) via `statsmodels.graphics.tsaplots`. Comment interpréter ces courbes ?

**Question 2.** Proposez deux hypothèses métiers pouvant expliquer la saisonnalité observée.

### 1.3 Décomposition

- Utilisez `statsmodels.tsa.seasonal_decompose` avec un modèle additif et une période de 52 (semaines).
- Affichez les composantes (tendance, saisonnalité, résidu) et commentez les périodes où la tendance change brusquement.

**Question 3.** D'après les résidus, la série est-elle parfaitement modélisée par une décomposition additif ? Justifiez.

## Partie 2 – Préparer un jeu supervisé

### 2.1 Découpage temporel

- Séparez les données en deux segments chronologiques : train (jusqu'à fin 1995) et test (à partir de 1996). Utilisez l'index
temporel pour garantir l'ordre.
- Créez une fonction `temporal_train_test_split` qui reçoit une série et une date de coupure, et retourne deux DataFrames.

**Question 4.** Pourquoi ne doit-on pas mélanger ou permuter les observations comme dans un `train_test_split` classique ?

### 2.2 Variables retardées et fenêtres glissantes

- Construisez un DataFrame `features` contenant :
  - la valeur de `ppm` retardée de 1, 2, 3 et 12 semaines ;
  - une moyenne mobile sur 4 semaines (`rolling(4).mean()`) ;
  - un indicateur de mois (`index.month`), encodé en sin/cos pour respecter la circularité.
- Définissez la variable cible comme la valeur de `ppm` à horizon 1 semaine (prévision `t+1`).

**Question 5.** Quels types d'informations apportent les retards 1 et 12 ? Pourquoi ajouter la moyenne mobile ?

### 2.3 Jeu de validation

- Conservez les 12 derniers mois du segment d'entraînement comme jeu de validation.
- Mettez en place un `TimeSeriesSplit` (scikit-learn) à 5 plis pour comparer la validation simple avec une validation croisée
glissante.

**Question 6.** Comparez les scores obtenus via la validation simple et via `TimeSeriesSplit`. Quelle méthode vous semble la plus
fiable dans ce contexte ?

## Partie 3 – Modèles de prévision

### 3.1 Baseline de persistance

- Implémentez une baseline `persistance` : la prévision de `t+1` est simplement la valeur observée à `t`.
- Calculez MAE et RMSE sur validation et sur test.

**Question 7.** Pourquoi est-il pertinent de comparer les modèles plus sophistiqués à cette baseline ?

### 3.2 Régression supervisée avec scikit-learn

- Entraînez un `RandomForestRegressor` et un `XGBRegressor` (si `xgboost` est installé). Pensez à restreindre la profondeur et à
valoriser `n_estimators`.
- Comparez leurs performances (MAE, RMSE) en validation croisée `TimeSeriesSplit`.
- Sélectionnez le meilleur modèle, ré-entrainez-le sur train+validation, puis évaluez-le sur le jeu de test.

**Question 8.** Comparez les erreurs obtenues à celles de la baseline. Quelles conclusions tirez-vous ?

### 3.3 Modèle ARIMA

- Utilisez `statsmodels.tsa.arima.model.ARIMA` avec un ordre $(p, d, q)$ inspiré des observations ACF/PACF.
- Ajustez les paramètres pour minimiser l'AIC (ou utilisez `auto_arima` si `pmdarima` est disponible).
- Confrontez les performances d'ARIMA avec celles du meilleur modèle supervisé.

**Question 9.** Quels avantages et limites constatez-vous entre l'approche « boîte noire » type forêt aléatoire et l'approche
statistique ARIMA ?

### 3.4 Prévision multi-horizon (optionnel)

- Étendez votre pipeline pour prédire la concentration à horizon 4 semaines (`t+4`), soit avec des modèles directs (multi-sorties),
soit par itération (prévision à horizon 1 répétée). Comparez les performances.

## Pour aller plus loin

- Testez la méthode `Prophet` de Meta (si installée) et comparez ses résultats.
- Expérimentez l'apprentissage profond (réseaux LSTM ou Temporal Convolutional Networks) sur ce jeu de données.
- Intégrez un contrôle de dérive conceptuelle : détectez les ruptures de tendance, ré-entraînez vos modèles et comparez.

Bon TP !
