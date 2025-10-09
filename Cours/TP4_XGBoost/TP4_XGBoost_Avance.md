# TP 4 : XGBoost avancé et gestion d'un ensemble de validation

## Objectifs pédagogiques

- Comprendre l'intérêt d'un ensemble de validation lors de l'entraînement d'un modèle XGBoost.
- Expliquer le rôle du `learning_rate` et savoir quand l'ajuster.
- Mettre en œuvre le suivi des performances avec `eval_set` et l'arrêt anticipé (`early stopping`).
- Ajuster les hyperparamètres (notamment `learning_rate` et `n_estimators`) pour tirer parti du critère d'arrêt anticipé.
- Appliquer ces notions sur le dataset Breast Cancer Wisconsin utilisé dans le TP précédent.

## Comprendre le `learning_rate`

Le `learning_rate` (aussi appelé **eta**) détermine l'amplitude du pas de gradient effectué à chaque itération. Plus il est petit,
plus les nouveaux arbres corrigent faiblement les erreurs des arbres précédents. À l'inverse, un taux élevé fait réaliser de
grands sauts dans l'espace des solutions.

Le réglage par défaut fourni par XGBoost est souvent satisfaisant : il constitue un bon compromis entre vitesse d'apprentissage
et stabilité. Deux situations courantes justifient cependant un ajustement :

1. **Convergence trop lente** : si, en observant les logs (`verbose=True`), la métrique met longtemps à s'améliorer ou si le
   modèle nécessite de nombreuses itérations avant de se stabiliser, c'est le signe que l'on apprend trop peu à chaque itération.
   On peut alors **augmenter légèrement le `learning_rate`** pour accélérer la convergence.
2. **Arrêt très rapide sans gain** : si l'entraînement se termine presque instantanément, par exemple en atteignant la limite
   d'`early_stopping_rounds` (15 itérations dans l'exemple), c'est que les mises à jour sont trop brusques. Le modèle ne parvient
   pas à faire mieux que les premières itérations. Il faut alors **réduire le `learning_rate`** pour réaliser des pas plus fins.

L'ajustement du `learning_rate` va toujours de pair avec celui du nombre d'arbres (`n_estimators`). Un taux plus faible nécessite
plus d'arbres pour converger, tandis qu'un taux plus fort en demande moins.

## Pourquoi utiliser un ensemble de validation ?

Lorsqu'on entraîne un modèle, il est tentant d'évaluer les performances uniquement sur l'ensemble d'entraînement. Pourtant, cette approche conduit souvent à du **sur-apprentissage (overfitting)** : le modèle mémorise trop finement les données d'entraînement et généralise mal sur de nouvelles données.

Un ensemble de validation permet de suivre les performances du modèle sur des données **jamais vues pendant l'ajustement des paramètres**. Pour XGBoost, cela se traduit par la possibilité d'observer l'évolution d'une métrique (perte logarithmique, erreur, AUC, etc.) sur cette validation et de stopper l'entraînement lorsqu'elle cesse de s'améliorer.

### Bénéfices clés

- **Détection précoce de l'overfitting** : dès que les performances sur validation se dégradent, on arrête l'entraînement.
- **Gain de temps** : inutile d'aller jusqu'au nombre maximal d'arbres si la qualité n'augmente plus.
- **Choix automatique du meilleur modèle** : XGBoost conserve les poids correspondant à la meilleure itération de validation.

## Mise en place pratique dans XGBoost

Voici les étapes typiques pour introduire un ensemble de validation avec la classe `XGBClassifier` (API `scikit-learn`) :

1. **Découpage des données** :
   - Séparer d'abord les données en train/test pour l'évaluation finale.
   - Créer ensuite un sous-ensemble de validation à partir du train (par exemple 80 % train / 20 % validation).
2. **Choisir un duo `learning_rate` / `n_estimators` cohérent** :
   - Partir du `learning_rate` par défaut (`0.1`) est une bonne pratique. On le diminue si l'apprentissage est instable ou trop
     rapide, on l'augmente s'il progresse trop lentement.
   - L'arrêt anticipé ne peut fonctionner que s'il dispose d'un nombre d'arbres suffisant à parcourir. Pour des taux faibles, on
     pousse `n_estimators` entre 800 et 2000 pour laisser de la marge.
3. **Appeler `fit` avec `eval_set` et `early_stopping_rounds`** :
   - `eval_set=[(X_train, y_train), (X_val, y_val)]` permet de surveiller plusieurs jeux de données.
   - `early_stopping_rounds=20` stoppe l'entraînement si la métrique ne s'améliore plus pendant 20 itérations consécutives.
   - Surveiller les logs (`verbose=True`) permet de juger de la vitesse de convergence et donc d'adapter le `learning_rate`.
4. **Choisir la métrique** :
   - Via `eval_metric` (`logloss`, `auc`, `error`, ...).

### Exemple de code complet

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Exemple avec le dataset Breast Cancer Wisconsin (scikit-learn)
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target

# 1. split train/test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# 3. définir un couple learning_rate / n_estimators cohérent
model = XGBClassifier(
    n_estimators=1000,      # grande valeur
    learning_rate=0.05,     # plus petit que le défaut pour des pas plus fins
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="binary:logistic"
)

# 4. entraînement avec suivi sur validation
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric="logloss",
    early_stopping_rounds=20,
    verbose=True
)

# 5. évaluation finale sur le test set
preds = model.predict(X_test)
print("Accuracy test:", accuracy_score(y_test, preds))
print("Meilleur nombre d'arbres:", model.best_iteration + 1)
```

Quelques points essentiels à retenir :

- **`best_iteration`** fournit l'indice de l'arbre retenu (0-indexé) au moment du meilleur score de validation.
- **`n_estimators` doit être plus grand que `best_iteration`** ; si vous fixez une valeur trop faible (ex. 50 arbres), le critère d'arrêt anticipé ne pourra jamais s'activer même si le modèle n'est pas optimal.
- Vous pouvez enregistrer les métriques d'apprentissage via `evals_result = model.evals_result()` pour tracer les courbes.

### Visualiser la courbe d'apprentissage

```python
evals_result = model.evals_result()
train_logloss = evals_result["validation_0"]["logloss"]
val_logloss = evals_result["validation_1"]["logloss"]

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(train_logloss, label="Train logloss")
plt.plot(val_logloss, label="Validation logloss")
plt.axvline(model.best_iteration, color="r", linestyle="--", label="Best iteration")
plt.xlabel("Nombre d'arbres")
plt.ylabel("Logloss")
plt.title("Courbes d'apprentissage XGBoost")
plt.legend()
plt.show()
```

Cette visualisation permet de constater à quel moment la validation cesse de s'améliorer.

## Bonnes pratiques supplémentaires

- **Fixer `early_stopping_rounds`** entre 10 et 50 selon la taille du dataset.
- **Conserver un test set indépendant** que l'on n'utilise jamais pour l'arrêt anticipé.
- **Tester plusieurs métriques** : pour des classes déséquilibrées, `auc` ou `logloss` sont plus informatives que l'accuracy.
- **Adapter le `learning_rate` à la dynamique d'entraînement** : augmenter légèrement le taux si la convergence est trop lente,
  le diminuer si l'entraînement s'arrête très tôt sans gain.
- **Réduire le taux d'apprentissage** (`learning_rate`) lorsqu'on augmente `n_estimators`, afin d'obtenir une optimisation plus fine.

## Travaux pratiques

Dans ce TP, réutilisez le dataset **Breast Cancer Wisconsin** introduit dans le TP 3.

1. Mettez en place un découpage `train / validation / test` et entraînez un `XGBClassifier` avec un premier couple `(learning_rate=0.1, n_estimators=800)` et `early_stopping_rounds=20`. Observez la vitesse de convergence grâce au `verbose`.
2. Ajustez ensuite le `learning_rate` dans les deux sens :
   - augmentez-le (ex. `0.2`) pour accélérer la convergence si elle est trop lente ; notez l'effet sur le nombre d'itérations avant arrêt ;
   - diminuez-le (ex. `0.03`) pour éviter un arrêt prématuré si l'entraînement se termine trop vite sans amélioration.
3. Tracez la courbe de la métrique suivie pour les différents réglages et reliez les observations aux paramètres choisis (`learning_rate`, `early_stopping_rounds`).
4. Comparez les performances sur le test set entre les différentes configurations et avec un entraînement sans ensemble de validation. Quels écarts constatez-vous ?

Consignez vos observations (meilleure itération, évolution des métriques, conclusions sur l'impact de l'ensemble de validation) dans votre compte-rendu.
