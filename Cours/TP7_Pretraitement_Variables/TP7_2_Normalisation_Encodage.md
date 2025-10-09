# TP7.2 — Normalisation et encodage des variables

[⬅️ Retour au README](../../README.md)

## Positionnement pédagogique
- **Durée indicative** : 3 heures.
- **Niveau** : BUT3 Informatique.
- **Compétences visées** : choix d'un schéma de transformation adapté au type de données, mise en œuvre de pipelines, prévention des fuites de données.

## Objectifs d'apprentissage
1. Comprendre pourquoi normaliser/standardiser les variables numériques.
2. Choisir un encodage pertinent pour les variables catégorielles en fonction du modèle cible.
3. Construire des pipelines reproductibles combinant transformations et modèles de classification.

## Cadre de travail
- Dataset : Adult Income (`fetch_openml`).
- Séparer `X` et `y`, puis découper le dataset en `train`/`test` stratifiés (`train_test_split`).
- Conserver une partie du jeu de test *non transformée* jusqu'à l'évaluation finale (pas de `fit` sur le test !).

## Activité 1 — Cartographier les types de variables (20 min)
1. Créer un tableau récapitulatif listant pour chaque colonne : type (`numérique`/`catégorielle`), nombre de modalités (catégorielles), présence de valeurs manquantes.
2. Visualiser rapidement la distribution de 3 variables numériques (`histplot` ou `boxplot`) et de 3 variables catégorielles (`value_counts()` + diagramme en barres).
3. Déduire les transformations potentielles (besoin de normalisation ? encodage ordinal pertinent ?).

## Activité 2 — Choix et comparaison des scalers (45 min)
1. Construire trois pipelines identiques (prétraitement + régression logistique) qui ne diffèrent que par le scaler appliqué aux numériques :
   - `StandardScaler`
   - `MinMaxScaler`
   - `RobustScaler`
2. Évaluer via validation croisée (`StratifiedKFold`, 5 folds) les métriques `accuracy`, `f1`, `roc_auc`.
3. Analyser l'impact sur la stabilité des coefficients (`coef_`) de la régression.
4. Question de réflexion : quelles situations du monde réel justifient l'usage de chacun de ces scalers ?

> ✍️ *À consigner* : un tableau comparatif des scores moyens et écarts-types.

## Activité 3 — Encodage des catégories (60 min)
1. Implémenter trois stratégies :
   - `OneHotEncoder` (baseline).
   - `OrdinalEncoder` avec un ordre logique pour `education` (par exemple : du niveau le plus faible au plus élevé).
   - `TargetEncoder` (via `category_encoders`).
2. Pour `TargetEncoder`, mettre en place une validation rigoureuse :
   - Utiliser `KFoldTargetEncoder` ou envelopper l'encodeur dans un pipeline avec `ColumnTransformer` + validation croisée pour éviter les fuites.
3. Évaluer chaque stratégie avec deux modèles : régression logistique et k-NN (`KNeighborsClassifier`).
4. Comparer les performances et discuter des limites :
   - Explosion du nombre de colonnes (OneHot).
   - Risque de sur-apprentissage (TargetEncoder).
   - Perte d'information sur l'ordre réel (OrdinalEncoder mal configuré).

> 📊 *Livrable* : un graphique en barres comparant `f1` pour chaque couple (encodeur, modèle).

## Activité 4 — Pipelines avancés et bonnes pratiques (30 min)
1. Introduire `ColumnTransformer` imbriqué :
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.impute import SimpleImputer

   numeric_transformer = Pipeline([
       ("imputer", SimpleImputer(strategy="median")),
       ("scaler", StandardScaler()),
   ])

   categorical_transformer = Pipeline([
       ("imputer", SimpleImputer(strategy="most_frequent")),
       ("encoder", OneHotEncoder(handle_unknown="ignore")),
   ])

   preprocess = ColumnTransformer([
       ("num", numeric_transformer, numeric_features),
       ("cat", categorical_transformer, categorical_features),
   ])
   ```
2. Expliquer l'intérêt d'imputer *dans* le pipeline (pour éviter les fuites et assurer la reproductibilité).
3. Tester ce pipeline complet avec deux modèles : régression logistique et SVM linéaire (`LinearSVC`).
4. Mesurer les temps d'entraînement et discuter des différences de performances.

## Activité 5 — Impact sur XGBoost (25 min)
1. Comparer deux pipelines :
   - XGBoost avec seulement l'encodage `OneHotEncoder` (pas de scaler).
   - XGBoost avec encodage catégoriel natif (`enable_categorical=True`) en convertissant les colonnes en `category` puis en `int`.
2. Évaluer `roc_auc` et `average_precision`.
3. Discussion :
   - Quels avantages/inconvénients du support natif des catégories par XGBoost ?
   - Dans quel cas conserveriez-vous un encodage one-hot malgré tout ?

## Synthèse attendue
- Tableau final récapitulant pour chaque combinaison (scaler, encodeur, modèle) : temps de fit, nombre de features générées, `f1`, `roc_auc`.
- Commentaire (8 lignes) expliquant la combinaison retenue pour la suite du parcours.

## Ressources
- Documentation scikit-learn : [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- Bibliothèque `category_encoders` : [documentation officielle](https://contrib.scikit-learn.org/category_encoders/)
- Article : *Handling Categorical Data for Machine Learning* (KDNuggets).
