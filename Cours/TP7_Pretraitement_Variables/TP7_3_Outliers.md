# TP7.3 — Détection et traitement des outliers

## Positionnement pédagogique
- **Durée indicative** : 2h30.
- **Niveau** : BUT3 Informatique.
- **Compétences visées** : identification d'observations aberrantes, maîtrise d'algorithmes de détection, intégration dans un pipeline de machine learning.

## Objectifs d'apprentissage
1. Comprendre l'impact des outliers sur différentes familles de modèles.
2. Mettre en œuvre des techniques statistiques et algorithmiques de détection.
3. Choisir une stratégie de traitement (suppression, caping, transformation) adaptée au contexte métier.

## Préparation
1. Reprendre le dataset Adult Income et les séparations `train`/`test` préparées dans les TP précédents.
2. Conserver la liste des variables numériques (`numeric_features`).
3. Créer un notebook dédié pour consigner les analyses visuelles et statistiques.

## Activité 1 — Diagnostic exploratoire (30 min)
1. Calculer pour chaque variable numérique : moyenne, écart-type, médiane, quantiles (Q1, Q3).
2. Visualiser les distributions avec des boxplots et histogrammes (`seaborn` recommandé).
3. Identifier visuellement les valeurs extrêmes et consigner vos observations (variable concernée, nombre de points suspects, hypothèses métier).

> 🗒️ *Astuce* : pensez à travailler sur un échantillon réduit (ex. 5 000 lignes) pour accélérer les visualisations si besoin.

## Activité 2 — Méthodes statistiques (35 min)
1. Implémenter l'approche IQR (InterQuartile Range) :
   - Calculer `IQR = Q3 - Q1`.
   - Définir des seuils : `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.
   - Compter le pourcentage de lignes à l'extérieur de ces seuils pour `capital-gain`, `capital-loss`, `hours-per-week`.
2. Implémenter le z-score :
   - Standardiser les variables.
   - Marquer comme outliers les points dont `|z| > 3`.
3. Comparer IQR vs z-score : quelles variables sont sensibles à la méthode choisie ?

## Activité 3 — Détection automatique (45 min)
1. Tester trois algorithmes : `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`.
2. Pour chacun :
   - Travailler sur les variables numériques standardisées (`StandardScaler`).
   - Ajuster l'hyperparamètre principal (`contamination` ou `nu`).
   - Visualiser les scores/anomalies via un scatter plot (PCA 2D) ou un histogramme des scores.
3. Comparer les ensembles de points détectés par chaque méthode (utiliser des ensembles Python pour compter les intersections).
4. Discuter : quelle méthode semble la plus pertinente et pourquoi ? (vitesse, interprétabilité, sensibilité aux paramètres).

> ⚙️ *Conseil* : encapsuler le scaler et l'algorithme dans un pipeline scikit-learn pour éviter les fuites.

## Activité 4 — Stratégies de traitement (30 min)
1. Implémenter trois stratégies :
   - **Suppression** : retirer les lignes marquées comme outliers.
   - **Winsorisation** : tronquer les valeurs extrêmes aux bornes définies par IQR.
   - **Transformation logarithmique** : appliquer `np.log1p` sur `capital-gain` et `capital-loss`.
2. Pour chaque stratégie, ré-entraîner :
   - Une régression logistique (avec pipeline complet du TP7.2).
   - Un modèle XGBoost (paramètres par défaut).
3. Comparer `f1`, `roc_auc` et le temps d'entraînement.
4. Discuter : quelle stratégie maximise le compromis performance/stabilité ?

## Activité 5 — Sensibilité des modèles (20 min)
1. Mesurer l'impact des outliers sur :
   - Les coefficients d'une régression logistique (avant/après nettoyage).
   - Les profondeurs d'arbre apprises par XGBoost (`max_depth` effective via importance des arbres).
2. Documenter vos observations dans un tableau ou un court commentaire.

## Synthèse attendue
- Tableau récapitulatif : méthode de détection / pourcentage d'observations marquées / stratégie appliquée / impact sur les métriques.
- Recommandations (6 à 8 lignes) :
  - Quelle méthode privilégieriez-vous dans un contexte industriel ?
  - Comment surveiller la réapparition d'outliers en production ?
  - Quels tests automatiseriez-vous (ex. alerte si >X % d'outliers) ?

## Ressources
- Documentation scikit-learn : [Outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- Article : *Anomaly Detection Techniques in Python* (Towards Data Science).
- Tutoriel vidéo : *Isolation Forest Explained* (StatQuest).
