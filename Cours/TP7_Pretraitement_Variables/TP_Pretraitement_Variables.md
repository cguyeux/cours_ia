# TP : Prétraitement et sélection des variables

## Objectifs pédagogiques

- Comprendre les étapes clés de préparation des variables avant l'entraînement d'un modèle.
- Savoir sélectionner les variables pertinentes et réduire la dimension.
- Mettre en œuvre la normalisation des variables quantitatives et l'encodage des variables qualitatives.
- Identifier et traiter les observations aberrantes (outliers).
- Adapter les pipelines de classification aux jeux de données déséquilibrés.
- Discuter des impacts spécifiques sur des algorithmes comme XGBoost.

## Prérequis

- Python
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- Notions de base sur la validation croisée et les pipelines

## Dataset proposé

Le TP s'appuie sur le dataset **Adult Income** (UCI Census Income) disponible via `fetch_openml`. L'objectif est de prédire si le revenu annuel dépasse 50k$ à partir de variables socio-démographiques. Le dataset contient un mélange de variables numériques et catégorielles, ainsi qu'un déséquilibre de classes modéré.

```python
from sklearn.datasets import fetch_openml
import pandas as pd

adult = fetch_openml(name="adult", version=2, as_frame=True)
X = adult.data
y = adult.target
```

## Déroulé du TP

1. Exploration du dataset et identification des types de variables.
2. Sélection de variables explicatives pertinentes.
3. Normalisation des variables quantitatives et encodage des variables qualitatives.
4. Détection et traitement des outliers.
5. Stratégies pour les classifications déséquilibrées.
6. Comparaison de pipelines (logistic regression vs XGBoost) pour mesurer l'impact du prétraitement.
7. Synthèse et recommandations.

---

## Partie 1 — Sélection de variables explicatives

### Objectif

Identifier les variables les plus pertinentes pour expliquer la variable cible, réduire le bruit et améliorer la généralisation du modèle.

### Méthodes classiques

1. **Analyse univariée** : tester l'association de chaque variable avec la cible.
   - Variables numériques : `f_classif`, `mutual_info_classif`.
   - Variables catégorielles : `chi2` (après encodage non négatif).
2. **Méthodes basées sur un modèle** : `SelectFromModel` avec `RandomForestClassifier` ou `Lasso`.
3. **Méthodes séquentielles** : `SequentialFeatureSelector` (forward/backward) en utilisant une validation croisée.

### Exemple de code (sélection univariée)

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

selector = SelectKBest(score_func=mutual_info_classif, k=20)
model = LogisticRegression(max_iter=1000)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("select", selector),
    ("clf", model),
])

pipe.fit(X, y)
```

### À propos de XGBoost

- XGBoost gère implicitement la sélection de variables grâce aux pénalités de complexité et aux scores de gain lors des splits.
- Malgré tout, supprimer les variables bruyantes peut accélérer l'entraînement et améliorer la robustesse.
- Sur des datasets volumineux ou très corrélés, une sélection préalable reste recommandée.

### Exercices

1. Tester trois valeurs de `k` dans `SelectKBest` et comparer l'`accuracy` et le `f1-score` sur un set de validation.
2. Utiliser `SelectFromModel` avec un `RandomForestClassifier` et comparer les variables retenues avec celles de l'analyse univariée.
3. Mesurer l'impact de la sélection des variables sur le temps d'entraînement d'un modèle XGBoost.

---

## Partie 2 — Normalisation et encodage des variables

### Objectif

Mettre toutes les variables sur une échelle comparable et convertir les catégories en représentations numériques appropriées pour l'entraînement.

### Méthodes classiques

1. **Normalisation / standardisation** : `StandardScaler`, `MinMaxScaler`, `RobustScaler`.
2. **Encodage des qualitatives** :
   - `OneHotEncoder` pour les modèles linéaires ou les distances.
   - `OrdinalEncoder` si l'ordre des modalités est connu.
   - `TargetEncoder` (nécessite une validation stricte pour éviter les fuites).
3. **Pipelines intégrés** pour garantir la reproductibilité (`ColumnTransformer`).

### Exemple de code (pipeline complet)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

numeric_transformer = Pipeline([
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000)),
])

clf.fit(X, y)
```

### À propos de XGBoost

- XGBoost supporte nativement les variables numériques non normalisées grâce aux arbres de décision.
- Les variables catégorielles peuvent être gérées via `enable_categorical=True` (après les avoir converties en catégories encodées en entiers) mais l'encodage one-hot reste souvent plus performant.
- Sur les variables très scalées ou à grande amplitude, la normalisation n'est pas obligatoire pour XGBoost mais peut aider d'autres modèles (régression logistique, SVM, kNN) dans un pipeline comparatif.

### Exercices

1. Comparer les performances d'une régression logistique avec `StandardScaler` vs `MinMaxScaler`.
2. Construire deux pipelines : l'un avec `OneHotEncoder`, l'autre en utilisant l'option catégorielle native de XGBoost (`enable_categorical=True`). Discuter des résultats.
3. Tester un `TargetEncoder` (ex. via `category_encoders`) en respectant une validation croisée, et comparer avec `OneHotEncoder`.

---

## Partie 3 — Détection et traitement des outliers

### Objectif

Identifier les observations atypiques susceptibles de fausser l'entraînement ou les métriques d'évaluation.

### Méthodes classiques

1. **Statistiques univariées** : écart interquartile (IQR), z-score.
2. **Modèles de détection** : `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`.
3. **Visualisations** : boxplots, scatter plots, projections PCA.
4. **Stratégies de traitement** : suppression, winsorisation, transformation log, modèles robustes.

### Exemple de code (IsolationForest)

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.02, random_state=42)
outlier_flags = iso.fit_predict(X[numeric_features])  # -1 = outlier, 1 = normal

# Filtrer les outliers pour réentraîner un modèle
mask = outlier_flags == 1
X_clean = X.loc[mask]
y_clean = y.loc[mask]
```

### À propos de XGBoost

- Les arbres de décision sont relativement robustes aux outliers, surtout si les valeurs extrêmes ne dominent pas les splits.
- Toutefois, la présence d'outliers peut ralentir le processus de split ou biaiser les feuilles si l'arbre doit créer des seuils spécifiques.
- Sur des modèles linéaires, les outliers ont un effet bien plus prononcé ; comparer les performances avec et sans nettoyage peut éclairer les choix de pipeline.

### Exercices

1. Utiliser l'approche IQR pour repérer les valeurs aberrantes sur `capital-gain` et `hours-per-week`. Quel pourcentage d'observations est filtré ?
2. Mettre en place un pipeline où l'on retire les outliers détectés par `IsolationForest` avant l'entraînement d'une régression logistique. Comparer les métriques.
3. Évaluer l'impact du filtrage des outliers sur un modèle XGBoost : gain en performance ? variation du temps d'entraînement ?

---

## Partie 4 — Classification déséquilibrée (Unbalanced)

### Objectif

Adapter les modèles pour optimiser les performances lorsque les classes positives sont rares ou très minoritaires.

### Méthodes classiques

1. **Rééchantillonnage** :
   - Sur-échantillonnage (`RandomOverSampler`, `SMOTE`).
   - Sous-échantillonnage (`RandomUnderSampler`).
2. **Pondération des classes** : `class_weight="balanced"` pour les modèles scikit-learn.
3. **Seuil de décision ajusté** : optimisation du seuil via la courbe ROC ou PR.
4. **Métriques adaptées** : `f1`, `balanced_accuracy`, `roc_auc`, `average_precision`.

### Exemple de code (pipeline avec SMOTE)

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

imb_pipe = ImbPipeline([
    ("preprocess", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

imb_pipe.fit(X_train, y_train)
y_pred = imb_pipe.predict(X_test)
print(classification_report(y_test, y_pred))
```

### À propos de XGBoost

- XGBoost offre le paramètre `scale_pos_weight` pour pondérer la classe minoritaire (`scale_pos_weight = négatifs/positifs`).
- Le booster gère bien les jeux déséquilibrés si le paramètre est correctement réglé et si l'on surveille des métriques adaptées (`auc`, `aucpr`).
- Combiner le pondération avec un léger sur-échantillonnage peut améliorer la stabilité, mais attention au risque d'overfitting.

### Exercices

1. Comparer trois stratégies : `class_weight="balanced"`, `SMOTE` + classe équilibrée, et réglage de `scale_pos_weight` pour XGBoost. Quelle combinaison offre le meilleur `f1-score` ?
2. Tracer la courbe `precision-recall` d'une régression logistique entraînée avec `SMOTE`. Choisir un seuil de décision optimisé pour maximiser le `f1`.
3. Entraîner un modèle XGBoost sans aucun rééquilibrage, puis avec `scale_pos_weight`. Comparer les `auc` et `average_precision`.

---

## Synthèse et livrables

- Comparer deux pipelines complets :
  1. Prétraitement + sélection + logistic regression.
  2. Prétraitement minimal + XGBoost avec réglage de `scale_pos_weight`.
- Rédiger une synthèse (10 lignes) discutant :
  - des choix de sélection de variables,
  - de l'impact de la normalisation/encodage,
  - de la gestion des outliers,
  - de la stratégie pour les classes déséquilibrées,
  - de l'apport (ou non) d'XGBoost.
- Fournir un notebook Jupyter documenté comprenant le code, les graphiques et les réponses aux questions.

Bon TP !
