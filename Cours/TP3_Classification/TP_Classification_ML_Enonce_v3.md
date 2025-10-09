# TP : Classification en Machine Learning

[⬅️ Retour au README](../../README.md)

## Algorithmes de classification utilisés

Avant de commencer le TP, nous allons présenter les algorithmes qui seront utilisés. Chaque méthode a ses forces, ses faiblesses et ses paramètres clés à optimiser. Des schémas et images sont fournis (liens ou générés automatiquement) pour aider à la compréhension.

### Decision Tree (Arbre de décision)

Principe : un arbre de décision segmente l’espace des variables par une série de questions du type « la valeur de feature X est-elle > à un seuil ? ». Chaque nœud correspond à une décision et les feuilles correspondent aux classes prédites.

- **Avantages** : interprétable, simple à visualiser, peu de preprocessing nécessaire.
- **Inconvénients** : tendance à sur-apprendre (overfitting) si l’arbre est trop profond.

**Hyperparamètres importants :**

- `max_depth` : limite la profondeur de l’arbre.
- `min_samples_split` : nombre minimal d’échantillons pour diviser un nœud.
- `min_samples_leaf` : nombre minimal d’échantillons dans une feuille.
- `criterion` : mesure d’impureté (`entropy`, `gini`).

<https://www.youtube.com/watch?v=ZVR2Way4nwQ>

### Random Forest

Principe : une forêt aléatoire est composée de nombreux arbres de décision. Chaque arbre est construit sur un sous-échantillon des données et un sous-ensemble de variables. La prédiction finale est obtenue par vote majoritaire.

- **Avantages** : robuste, réduit l’overfitting par rapport à un seul arbre, performant sans tuning complexe.
- **Inconvénients** : moins interprétable qu’un arbre unique, plus coûteux en calcul.

**Hyperparamètres importants :**

- `n_estimators` : nombre d’arbres.
- `max_depth` : profondeur maximale des arbres.
- `max_features` : nombre de variables prises en compte à chaque split.
- `min_samples_leaf` : nombre minimal d’échantillons dans une feuille.

<https://www.youtube.com/watch?v=v6VJ2RO66Ag>

### XGBoost (eXtreme Gradient Boosting)

Principe : le boosting construit les arbres séquentiellement. Chaque nouvel arbre corrige les erreurs des arbres précédents. XGBoost est une implémentation optimisée et très utilisée du gradient boosting.

- **Avantages** : souvent l’algorithme le plus performant sur données tabulaires.
- **Inconvénients** : tuning plus complexe, temps d’entraînement parfois élevé.

**Hyperparamètres importants :**

- `n_estimators` : nombre d’arbres.
- `learning_rate` : taux d’apprentissage (impact de chaque arbre).
- `max_depth` : profondeur maximale.
- `subsample` : fraction d’échantillons utilisés par arbre.
- `colsample_bytree` : fraction de variables utilisées par arbre.

## Principe du `train_test_split`

Lorsqu’on entraîne un modèle de machine learning, il est essentiel de pouvoir évaluer ses performances sur des données jamais vues. On sépare donc le dataset en deux parties :

- **train set** : utilisé pour l’apprentissage du modèle.
- **test set** : utilisé uniquement pour l’évaluation finale.

Exemple typique : 80 % pour l’entraînement et 20 % pour le test, en respectant la proportion des classes (stratification).

<https://www.youtube.com/watch?v=SjOfbbfI2qY>

## Métriques d’évaluation en classification

Pour juger la qualité d’un modèle de classification, plusieurs métriques sont utilisées :

- **Accuracy** : proportion de prédictions correctes.
- **Precision** : parmi les prédictions positives, combien sont réellement positives ?
- **Recall (sensibilité)** : parmi les vrais positifs, combien sont retrouvés par le modèle ?
- **F1-score** : moyenne harmonique de la précision et du rappel, utile quand les classes sont déséquilibrées.

<https://www.youtube.com/watch?v=Kdsp6soqA7o>

## Objectifs pédagogiques

- Comprendre et mettre en pratique plusieurs algorithmes de classification : Decision Tree, Random Forest, XGBoost.
- Savoir préparer un dataset, entraîner, évaluer et comparer des modèles.
- Analyser et interpréter des résultats (métriques, matrices de confusion, importances de variables).

## Prérequis

- Python
- `pandas`
- `numpy`
- `matplotlib`
- Notions de `scikit-learn`

## Dataset utilisé

Le TP utilisera le dataset **Breast Cancer Wisconsin**, intégré à scikit-learn. Il s’agit de prédire si une tumeur est bénigne ou maligne à partir de mesures de cellules.

## Déroulé du TP

Pour vous aider à progresser pas à pas, le TP est découpé en étapes opérationnelles. Chaque étape est l’occasion d’ajouter une
nouvelle cellule à votre notebook et d’en commenter les résultats.

1. **Préparer l’environnement de travail**
   - Créez un nouveau notebook ou dupliquez le modèle fourni puis importez les bibliothèques nécessaires (`pandas`, `numpy`,
     `matplotlib.pyplot`, `seaborn`, `sklearn`).
   - Chargez le dataset Breast Cancer depuis `sklearn.datasets` et transformez-le en `DataFrame` pour faciliter l’exploration.
   - Affichez la dimension des données, les noms de colonnes et les premières lignes pour vérifier le chargement.
2. **Explorer le dataset**
   - Recherchez des valeurs manquantes, examinez la distribution des classes et calculez quelques statistiques descriptives.
   - Visualisez au moins une paire de variables (via `pairplot` ou `scatterplot`) pour repérer d’éventuelles séparations naturelles.
   - Identifiez les corrélations fortes et notez les variables qui pourraient être redondantes.
3. **Mettre en place la validation**
   - Séparez les données en ensembles d’entraînement et de test avec `train_test_split`, en stratifiant sur la variable cible.
   - Normalisez les features (StandardScaler ou MinMaxScaler) si nécessaire. Conservez la version brute si vous souhaitez comparer.
   - Enregistrez la taille de chaque ensemble et justifiez le choix du ratio train/test.
4. **Établir un modèle de référence (Decision Tree)**
   - Entraînez un arbre de décision simple pour disposer d’un point de comparaison.
   - Calculez les métriques principales sur le train et le test, puis discutez rapidement de la qualité du modèle.
   - Affichez la matrice de confusion et, si possible, l’arbre ou les règles apprises.
5. **Améliorer les performances (Random Forest)**
   - Entraînez une Random Forest en commençant par les hyperparamètres par défaut.
   - Testez au moins deux configurations différentes (par exemple nombre d’arbres ou profondeur maximale) et consignez les
     résultats dans un tableau comparatif.
   - Analysez l’importance des variables fournie par le modèle et commentez les trois variables les plus influentes.
6. **Explorer un modèle plus avancé (XGBoost)**
   - Installez/importez `xgboost` si nécessaire et entraînez un premier modèle avec les paramètres standards.
   - Ajustez progressivement `learning_rate`, `max_depth` ou `n_estimators` en observant l’impact sur les métriques de test.
   - Documentez les temps d’entraînement ou les difficultés rencontrées (gestion des paramètres, convergence, etc.).
7. **Comparer et interpréter**
   - Rassemblez les métriques principales (accuracy, precision, recall, f1-score) pour les trois modèles dans un tableau
     synthétique et discutez des différences observées.
   - Représentez les matrices de confusion et soulignez les zones d’erreurs communes ou spécifiques à chaque algorithme.
   - Décrivez, à partir des importances de variables, les caractéristiques qui contribuent le plus à la décision.
8. **Conclure et ouvrir**
   - Rédigez un court paragraphe récapitulatif expliquant le modèle que vous retiendriez pour une mise en production et pourquoi.
   - Listez les axes d’amélioration possibles : réglages supplémentaires, ajout de validation croisée, gestion de classes
     déséquilibrées, interprétabilité avancée, etc.
   - Vérifiez que le notebook est clair (titres, commentaires, conclusions) et prêt pour la remise.

## Livrables

- Un notebook Jupyter complété (code + réponses textuelles aux questions).
- Un court compte rendu (5–10 lignes) comparant les modèles et concluant sur celui qui semble le plus adapté.
