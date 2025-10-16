# TP6 — Séries temporelles : parcours guidé

[⬅️ Retour au sommaire](../../LISEZMOI.md)

## Pourquoi ce parcours ?

Les données temporelles sont omniprésentes (IoT, mobilité, transport, énergie). Ce TP vous propose un **fil rouge progressif** :
partir d’une série brute, la comprendre, la stabiliser puis comparer plusieurs familles de modèles de prévision. Le tout est découpé
en quatre sous-TP autonomes (environ 1h30 à 2h chacun) pour s’adapter à votre disponibilité.

## Scénario métier

Vous assistez l’opérateur des vélos en libre-service de Washington DC. Votre mission : **anticiper la demande horaire** pour mieux
réguler la flotte. Le jeu de données de référence est `Bike_Sharing_Demand` (OpenML #42712), contenant les locations réelles entre
2011 et 2012.

## Fil conducteur

1. **Explorer** la série pour comprendre ses motifs (pics quotidiens, week-ends, saisons).
2. **Séparer** tendance, saisonnalité et bruit pour stabiliser le signal.
3. **Tester** des modèles sériels classiques (persistance, AR, ARIMA) et comprendre leurs limites.
4. **Comparer** avec des modèles supervisés utilisant des features retardées et des variables exogènes.

Chaque sous-TP se conclut par une mini-synthèse pour alimenter un rapport final.

## Organisation recommandée

- **Durée indicative** : 6 à 8 heures au total.
- **Environnement** : Python ≥ 3.9, `pandas`, `matplotlib`/`seaborn`, `statsmodels`, `scikit-learn`, `xgboost` (optionnel).
- **Livrable global** : un notebook ou rapport consolidant vos résultats + recommandations opérationnelles.

> 💡 Conseil : créez un dossier `notebooks/TP6` et un notebook par sous-TP afin de pouvoir revenir facilement sur vos analyses.

## Parcours en quatre sous-TP

1. [**TP6.1 — Exploration et visualisation de la demande**](TP6_1_Exploration_Visuelle.md)  
   Chargement du dataset, contrôles qualité, visualisations globales et cycliques, autocorrélations.

2. [**TP6.2 — Tendance, saisonnalité et stationnarité**](TP6_2_Decomposition_Stationnarite.md)  
   Décomposition additive/multiplicative, différenciations simple et saisonnière, tests ADF.

3. [**TP6.3 — Modèles AR et ARIMA**](TP6_3_Modeles_AR_ARIMA.md)  
   Baseline de persistance, sélection de l’ordre AR, entraînement ARIMA, analyse des résidus, prévisions court terme.

4. [**TP6.4 — Approches supervisées et comparaison finale**](TP6_4_Modeles_Supervises.md)  
   Création de features retardées, validation glissante, Random Forest / (X)GBoost, comparaison avec ARIMA.

## Livrables suggérés

- Tableau comparatif MAE/RMSE (baseline vs AR vs ARIMA vs modèle supervisé).
- Graphiques clés : décomposition, ACF/PACF, prévision vs réalité, importances de features.
- Recommandations pour l’opérateur (gestion de flotte, besoins en données exogènes, fréquence de réentraînement).

## Ressources utiles

- Documentation [`statsmodels` ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html).
- Guide [`scikit-learn` TimeSeriesSplit](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).
- Pour aller plus loin : `pmdarima.auto_arima`, SARIMAX, Prophet, réseaux LSTM/TCN.

Bon parcours ! N’hésitez pas à noter vos hypothèses et limites tout au long du TP : elles nourriront votre synthèse finale.
