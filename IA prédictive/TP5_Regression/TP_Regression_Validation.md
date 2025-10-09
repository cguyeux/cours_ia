# TP : Régression, validation croisée et contrôle de la complexité

[⬅️ Retour au README](../../README.md)

## Objectifs pédagogiques

- Comprendre la différence entre un problème de classification et un problème de régression.
- Mettre en place une régression supervisée et interpréter ses métriques.
- Introduire la validation croisée et son rôle dans la sélection de modèles.
- Explorer l'impact de l'hyperparamètre `max_depth` sur la complexité d'un arbre de décision.
- Savoir structurer un protocole expérimental en trois ensembles : entraînement, validation, test.

## Prérequis

- Python
- `pandas`
- `numpy`
- `matplotlib`
- Notions de `scikit-learn`

## Contexte

Dans les TP précédents, vous avez manipulé des tâches de **classification**, où l'objectif est de prédire une étiquette discrète
(benin/malin, oui/non, type de client, etc.). Dans ce TP, nous allons aborder la **régression** : le modèle doit prédire une valeur
continue (prix, température, consommation, durée, ...).

Nous utiliserons deux jeux de données :

1. **California Housing** (scikit-learn) : prédiction du prix médian des logements par district.
2. **DataFrame synthétique** avec une relation non linéaire entre les variables.

## Partie 1 – Classification vs Régression

### 1.1 Comparaison théorique

- **Classification** : sortie discrète, métriques type accuracy, F1-score.
- **Régression** : sortie continue, métriques type MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), $R^2$.

**Question 1.** Expliquez, à partir de votre expérience sur les TPs précédents, dans quels cas vous choisiriez la classification ou la
régression. Illustrez par un exemple métier.

### 1.2 Premiers pas en régression

Dans un notebook Jupyter, implémentez le code suivant pour charger le dataset California Housing et entraîner un modèle de
`DecisionTreeRegressor`.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Chargement du dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Découpage train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle de régression
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)

# Évaluation
preds = reg.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

print(f"MAE : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2 : {r2:.3f}")
```

**Question 2.** Ajoutez une cellule qui convertit les métriques en un petit DataFrame, puis commentez leur signification.

## Partie 2 – Validation croisée et ensembles de données

### 2.1 Pourquoi aller au-delà du simple train/test ?

Lorsque vous ajoutez de nouvelles variables (features) ou que vous ajustez des hyperparamètres, il est possible d'améliorer les
performances **par chance** sur le jeu de test. Cela signifie que votre modèle s'est adapté à ce jeu précis, mais ne généralisera
pas forcément. Pour éviter cela :

- **Train set** : utilisé pour ajuster les paramètres du modèle.
- **Validation set** : utilisé pendant l'entraînement pour savoir quand s'arrêter et éviter le sur-apprentissage.
- **Test set** : utilisé pour comparer différents modèles/hyperparamètres et choisir la meilleure configuration.

### 2.2 Validation croisée $k$-fold

La validation croisée découpe le train set en $k$ sous-ensembles. À tour de rôle, on utilise $k-1$ sous-ensembles pour entraîner et le
sous-ensemble restant pour valider. On répète l'opération $k$ fois et on moyenne les métriques.

**Code à compléter :**

```python
from sklearn.model_selection import KFold, cross_val_score

reg = DecisionTreeRegressor(max_depth=5, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    reg,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=kf
)

print("Scores RMSE (négatifs) :", scores)
print("RMSE moyen :", -scores.mean())
print("Écart-type :", scores.std())
```

**Question 3.** Expliquez pourquoi scikit-learn retourne des scores négatifs pour la RMSE. Comment convertir ces valeurs pour les
interpréter ?

**Question 4.** Comparez le RMSE moyen obtenu en validation croisée avec celui calculé sur le test set sans CV. Que constatez-vous ?

### 2.3 Sélection de modèle via validation croisée

Nous allons comparer trois modèles :

- `DecisionTreeRegressor`
- `RandomForestRegressor`
- `GradientBoostingRegressor`

**Exercice guidé :**

1. Créez une fonction `evaluate_model(model, X, y)` qui retourne le RMSE moyen en validation croisée (5 folds).
2. Appliquez-la aux trois modèles avec leurs paramètres par défaut.
3. Rangez les résultats dans un DataFrame comparatif.
4. Sélectionnez le modèle offrant le meilleur RMSE moyen et ré-entraînez-le sur le train set complet.
5. Évaluez-le sur le test set et commentez la cohérence entre validation croisée et test.

## Partie 3 – Focus sur `max_depth`

### 3.1 Comprendre la complexité d'un arbre

L'hyperparamètre `max_depth` limite le nombre de niveaux d'un arbre de décision. Un arbre trop profond mémorise les données
(sur-apprentissage), tandis qu'un arbre trop peu profond ne capte pas la structure (sous-apprentissage).

**Exercice :**

```python
results = []
for depth in range(2, 11):
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    rmse_cv = -cross_val_score(
        reg,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=5
    ).mean()
    results.append({"max_depth": depth, "rmse_cv": rmse_cv})

results_df = pd.DataFrame(results)
print(results_df)
```

**Question 5.** Tracez `max_depth` vs `rmse_cv`. À partir de quelle profondeur le RMSE se stabilise-t-il ?

### 3.2 Visualiser sur un dataset synthétique

Créez un dataset artificiel avec une relation non linéaire (ex. `y = sin(x) + bruit`). Utilisez `DecisionTreeRegressor` avec
`max_depth` croissant pour observer visuellement le sous-apprentissage et le sur-apprentissage.

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)
X_syn = np.sort(5 * rng.rand(200, 1), axis=0)
y_syn = np.sin(X_syn).ravel()
y_syn += 0.5 * (rng.rand(200) - 0.5)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, depth in zip(axes.ravel(), [2, 4, 6, 10]):
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    reg.fit(X_syn, y_syn)
    ax.scatter(X_syn, y_syn, s=10, label="Données")
    ax.plot(X_syn, reg.predict(X_syn), color="red", label="Prédiction")
    ax.set_title(f"max_depth = {depth}")
    ax.legend()

plt.tight_layout()
plt.show()
```

**Question 6.** Commentez les courbes obtenues. Quel compromis entre biais et variance observez-vous ?

### 3.3 Bonnes pratiques

- Commencez toujours par tester différentes valeurs de `max_depth` (ex. 2 à 10) pour calibrer la complexité.
- Surveillez le RMSE (ou une autre métrique) sur l'ensemble de validation : si la performance se dégrade après une certaine profondeur, c'est un signe de sur-apprentissage.
- Utilisez la validation croisée pour robustifier votre choix : une seule séparation train/test peut être trompeuse.

## Partie 4 – Synthèse et prolongements

1. Rédigez un court paragraphe expliquant comment la régression se différencie de la classification, en mentionnant les métriques, les risques et les outils utilisés dans ce TP.
2. Expliquez comment la validation croisée vous aide à décider d'ajouter ou non une nouvelle variable ou de modifier un hyperparamètre.
3. Décrivez votre stratégie pour déterminer la bonne valeur de `max_depth` sur un nouveau projet.

## Ressources supplémentaires

- Documentation scikit-learn – [Régression](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- Article Towards Data Science – [Understanding Cross-Validation](https://towardsdatascience.com/cross-validation-70289113a072)
- Documentation scikit-learn – [`DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

**Livrable attendu :** un notebook Jupyter contenant le code, les réponses aux questions et une conclusion synthétique sur les choix de
modélisation.
