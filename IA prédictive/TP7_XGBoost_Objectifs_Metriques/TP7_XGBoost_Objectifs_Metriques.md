# TP 7 : XGBoost – Fonctions objectives et métriques

[⬅️ Retour au sommaire](../../LISEZMOI.md)

## Objectifs pédagogiques

- Comprendre le rôle de la fonction objective dans XGBoost et savoir la choisir selon le problème.
- Identifier les fonctions objectives disponibles pour des cas particuliers (Poisson, Tweedie, etc.).
- Choisir et combiner des métriques d'entraînement, de validation et d'interprétation humaine.
- Définir une fonction objective et une métrique personnalisées pour répondre à une contrainte métier.
- Mettre en pratique ces concepts sur des scénarios de classification et de régression.

## 1. Rôle de la fonction objective

La **fonction objective** (ou fonction de perte) est la quantité que XGBoost cherche à minimiser. Elle combine :

1. Une **perte de formation** (par exemple l'erreur quadratique pour la régression).
2. Un **terme de régularisation** (L1/L2) qui pénalise la complexité des arbres.

Lors de chaque itération, XGBoost ajoute un arbre qui diminue la valeur de cette fonction. Le choix de l'objective est donc crucial : il détermine la manière dont le modèle mesure l'erreur et adapte les mises à jour de gradient.

### 1.1 Valeur par défaut selon l'API

| API XGBoost | Estimator | Objective par défaut |
|-------------|-----------|-----------------------|
| `xgboost.XGBClassifier` | Binaire (`y` à 2 classes) | `binary:logistic` (log-loss avec sortie sigmoïde)
| `xgboost.XGBClassifier` | Multiclasse (`num_class > 2`) | `multi:softprob` (log-loss multi-classe avec probabilités)
| `xgboost.XGBRegressor`  | Régression | `reg:squarederror` (MSE)
| `xgboost.train` | Booster `gbtree` | `reg:squarederror` si non spécifié |

### 1.2 Tour d'horizon des fonctions objectives utiles

| Type de tâche | Objective | Description | Quand l'utiliser ? |
|---------------|-----------|-------------|---------------------|
| Classification binaire | `binary:logistic` | Log-loss binaire, sortie comprise entre 0 et 1. | Cas standard : churn, fraude, diagnostic binaire. |
| Classification binaire | `binary:logitraw` | Log-loss binaire mais sortie non transformée (log-odds). | Lorsque l'on souhaite contrôler soi-même la transformation sigmoïde ou appliquer un seuil non standard. |
| Classification multi-classe | `multi:softprob` | Log-loss multi-classe, sortie de probabilités pour chaque classe. | Lorsque l'on souhaite obtenir des probabilités complètes. |
| Classification multi-classe | `multi:softmax` | Log-loss multi-classe, sortie de la classe prédite directement. | Lorsque seules les classes finales nous intéressent (perte de probas). |
| Régression | `reg:squarederror` | Erreur quadratique moyenne. | Régression classique, valeurs continues. |
| Régression robuste | `reg:absoluteerror` | Erreur absolue moyenne (MAE). | Quand les valeurs aberrantes sont nombreuses et qu'on souhaite une perte moins sensible aux outliers. |
| Comptage | `count:poisson` | Log-likelihood Poisson. | Comptage d'évènements rares, nombre de visites, nombre d'incidents. |
| Assurance / énergie | `reg:tweedie` | Tweedie deviance (entre Poisson et Gamma). | Modélisation de montants avec forte masse en zéro + queue positive (sinistres). |
| Classement | `rank:pairwise`, `rank:ndcg`, ... | Pertes inspirées du ranking. | Recommandation, moteurs de recherche. |
| Survie | `survival:cox` | Modèle de Cox (données censurées). | Analyse de survie, temps avant événement. |

#### Focus sur Poisson et Tweedie

- **`count:poisson`** : la cible doit être un entier positif. XGBoost applique une transformation exponentielle en sortie, ce qui impose un apprentissage sur les log-comptes. Indispensable lorsque la variance des comptes croît avec la moyenne.
- **`reg:tweedie`** : couvre un continuum entre Poisson (`power=1`), Gamma (`power=2`) et des distributions intermédiaires (`1 < power < 2`). Pratique pour les montants d'assurance (beaucoup de zéros + queue longue). Nécessite de définir `tweedie_variance_power`.

```python
from xgboost import XGBRegressor

modele_poisson = XGBRegressor(
    objective="count:poisson",
    max_depth=4,
    tree_method="hist",
)

modele_tweedie = XGBRegressor(
    objective="reg:tweedie",
    tweedie_variance_power=1.5,
    max_depth=4,
    tree_method="hist",
)
```

## 2. Bien choisir sa métrique d'évaluation

La métrique (`eval_metric`) permet de suivre la performance au cours de l'entraînement sur l'ensemble d'entraînement et sur les ensembles de validation fournis dans `eval_set`. Par défaut, XGBoost choisit une métrique cohérente avec l'objective, mais il est souvent pertinent de :

1. **Suivre une métrique métier** compréhensible (MAE, F1-score, etc.) sur la validation pour communiquer aux parties prenantes.
2. **Utiliser une métrique optimisée par XGBoost** (log-loss, RMSE…) pour l'arrêt anticipé (`early_stopping_rounds`).

### 2.1 Exemple : classification déséquilibrée

Supposons un cas de fraude où nous privilégions le rappel (Recall). Nous pouvons :

- Optimiser l'objective par défaut (`binary:logistic`).
- Utiliser `eval_metric=["auc", "logloss"]` pour guider l'entraînement.
- Calculer à chaque epoch une métrique métier (par exemple F1) à l'aide d'un callback ou en post-traitement via `model.evals_result_`.

```python
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

model = XGBClassifier(
    scale_pos_weight=10,
    eval_metric=["auc", "logloss"],
    early_stopping_rounds=30,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)

# Calcul d'une métrique métier sur la validation
val_preds = model.predict(X_val)
print("F1 validation:", f1_score(y_val, val_preds))
```

### 2.2 Exemple : régression avec MAE journalisé

```python
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

reg = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    objective="reg:squarederror",
    eval_metric=["rmse"],  # métrique utilisée pour l'arrêt anticipé
    early_stopping_rounds=30,
)

reg.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)

val_preds = reg.predict(X_val)
print("MAE validation (métrique métier):", mean_absolute_error(y_val, val_preds))
```

Ici, XGBoost surveille `rmse` (plus adaptée à l'objective MSE) pour l'arrêt anticipé, tandis qu'on journalise un `MAE` à destination des décideurs.

### 2.3 Métriques multiples et callbacks

On peut fournir une **liste** à `eval_metric`. XGBoost journalise alors toutes les métriques mais n'utilise que la première pour l'arrêt anticipé. Après l'entraînement, `model.evals_result()` retourne un dictionnaire qui contient, pour chaque jeu de données, toutes les courbes des métriques suivies.

Pour des métriques qui n'existent pas nativement (par exemple un score métier interne), on peut écrire un **callback personnalisé** dérivé de `xgboost.callback.TrainingCallback` :

```python
import numpy as np
import xgboost as xgb

mae_history = []

class LogMAE(xgb.callback.TrainingCallback):
    def __init__(self, dval, y_val):
        self.dval = dval
        self.y_val = y_val

    def after_iteration(self, model, epoch, evals_log):
        preds = model.predict(self.dval)
        mae = float(np.mean(np.abs(preds - self.y_val)))
        mae_history.append(mae)
        print(f"[epoch {epoch}] validation-mae(humain)={mae:.4f}")
        return False  # continuer l'entraînement

params = {"objective": "reg:squarederror", "eta": 0.05}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

callbacks = [LogMAE(dval, y_val)]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dtrain, "train"), (dval, "validation")],
    callbacks=callbacks,
)
```

Ici, `LogMAE` calcule et journalise une métrique interprétable tout en laissant XGBoost optimiser sa métrique principale (`rmse`, `logloss`, etc.).

## 3. Fonctions objectives et métriques personnalisées

Il peut arriver qu'une contrainte métier ne soit pas couverte par les objectifs standard. On peut alors définir :

- une **fonction objective personnalisée** qui fournit gradient et hessien,
- une **métrique personnalisée** pour mesurer la performance selon un critère spécifique.

### 3.1 Syntaxe générale avec `xgb.train`

```python
import xgboost as xgb

def custom_objective(preds, dtrain):
    # preds : sorties brutes du booster (avant transformation)
    # dtrain : DMatrix contenant les labels
    labels = dtrain.get_label()
    grad = ...  # dérivée première
    hess = ...  # dérivée seconde
    return grad, hess

def custom_metric(preds, dtrain):
    labels = dtrain.get_label()
    metric_value = ...
    return "nom_metric", metric_value

params = {
    "max_depth": 4,
    "eta": 0.05,
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    obj=custom_objective,
    feval=custom_metric,
    evals=[(dtrain, "train"), (dval, "validation")],
    early_stopping_rounds=30,
)
```

### 3.2 Exemple : contraindre la variabilité des prédictions

**Objectif** : faire en sorte que les prédictions aient la même variance que la cible. On définit une perte qui pénalise l'écart entre la variance des prédictions et celle de la cible.

Soit \( \sigma_y^2 \) la variance de la cible et \( \sigma_{\hat{y}}^2 \) la variance des prédictions. Nous minimisons :

\[
\mathcal{L} = \big( \sigma_{\hat{y}}^2 - \sigma_y^2 \big)^2.
\]

Le gradient par rapport à la prédiction \(\hat{y}_i\) se calcule :

\[
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{4}{n} \big( \sigma_{\hat{y}}^2 - \sigma_y^2 \big) (\hat{y}_i - \overline{\hat{y}}),
\]

où \( n \) est le nombre d'échantillons et \(\overline{\hat{y}}\) la moyenne des prédictions.

Pour le hessien, on utilise une approximation constante positive afin de stabiliser l'entraînement.

```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Préparer les données
X, y = fetch_california_housing(return_X_y=True, as_frame=False)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

sigma_y2 = np.var(y_train)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

def variance_matching_obj(preds, dtrain):
    labels = dtrain.get_label()
    n = preds.size
    mean_pred = np.mean(preds)
    var_pred = np.mean((preds - mean_pred) ** 2)
    grad = (4.0 / n) * (var_pred - sigma_y2) * (preds - mean_pred)
    hess = np.full_like(preds, 4.0 / n)  # approximation positive
    return grad, hess

def variance_ratio_metric(preds, dtrain):
    labels = dtrain.get_label()
    var_pred = np.var(preds)
    var_label = np.var(labels)
    ratio = var_pred / (var_label + 1e-12)
    return "var_ratio", ratio

params = {
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",  # utilisé pour initialiser le booster
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=400,
    obj=variance_matching_obj,
    feval=variance_ratio_metric,
    evals=[(dtrain, "train"), (dval, "validation")],
    early_stopping_rounds=30,
)
```

**Interprétation :**

- `variance_matching_obj` pousse la variance des prédictions vers celle des cibles.
- `variance_ratio_metric` journalise le ratio \( \sigma_{\hat{y}}^2 / \sigma_y^2 \). On attend un ratio proche de 1 sur la validation.
- On peut combiner cette objective avec une perte standard via un mélange (par exemple \( \mathcal{L}_{total} = \text{MSE} + \lambda \mathcal{L}_{variance} \)) en ajoutant le gradient de la MSE.

### 3.3 Variante : mélange avec la MSE

```python
def blended_objective(preds, dtrain, lam=0.1):
    labels = dtrain.get_label()
    n = preds.size
    mean_pred = np.mean(preds)
    var_pred = np.mean((preds - mean_pred) ** 2)
    # composante variance
    grad_var = (4.0 / n) * (var_pred - sigma_y2) * (preds - mean_pred)
    hess_var = np.full_like(preds, 4.0 / n)
    # composante MSE
    residuals = preds - labels
    grad_mse = residuals
    hess_mse = np.ones_like(preds)
    grad_total = grad_mse + lam * grad_var
    hess_total = hess_mse + lam * hess_var
    return grad_total, hess_total
```

Cette approche permet de garder de bonnes performances en MSE tout en contraignant la variabilité des prédictions.

## 4. Travaux pratiques

### Exercice 1 – Explorer les objectives standards

1. Reprenez le dataset Breast Cancer Wisconsin (classification binaire).
2. Entraînez successivement trois modèles :
   - `objective="binary:logistic"`, `eval_metric=["logloss"]` ;
   - `objective="binary:logitraw"`, `eval_metric=["auc", "error"]` ;
   - `objective="rank:pairwise"` avec `eval_metric=["auc"]` (observez les différences de convergence).
3. Comparez les performances (AUC, accuracy) et notez les effets sur la calibration des probabilités.

### Exercice 2 – Choisir une métrique métier

1. Sur un problème de régression (California Housing), entraînez un `XGBRegressor` avec `objective="reg:squarederror"`.
2. Suivez en parallèle `rmse` (pour l'arrêt anticipé) et `mae` (métrique métier) via `eval_metric=["rmse", "mae"]`.
3. Tracez les courbes `rmse` et `mae` pour le train et la validation et discutez de l'intérêt de chaque métrique.

### Exercice 3 – Ajuster une objective Poisson ou Tweedie

1. Construisez un dataset de comptage (par exemple nombre de locations de vélos par heure) ou utilisez un jeu open data.
2. Comparez les objectifs `reg:squarederror`, `count:poisson` et `reg:tweedie` (`tweedie_variance_power=1.3`).
3. Analysez l'impact sur les prédictions (positivité, dispersion) et sur les métriques `rmse`, `mae`, `mean_poisson_deviance`.

### Exercice 4 – Implémenter l'objectif « variance égale »

1. Suivez l'exemple de la section 3.2 pour définir `variance_matching_obj` et `variance_ratio_metric`.
2. Mesurez le ratio de variance sur validation et comparez-le à un modèle standard (`reg:squarederror`).
3. Testez différentes valeurs de `lam` dans la version mélangée (section 3.3) pour équilibrer précision et variabilité.

### Exercice 5 – Journaliser une métrique custom

1. Créez une fonction `feval` qui renvoie la `MAPE` (Mean Absolute Percentage Error) sur l'ensemble de validation.
2. Intégrez-la dans `xgb.train` tout en conservant `rmse` comme métrique d'arrêt anticipé.
3. Exportez les courbes `rmse` et `MAPE` et discutez de l'interprétabilité des métriques pour un public non technique.

## Points clés à retenir

- L'objective par défaut est cohérente avec la tâche mais peut être modifiée pour des besoins spécifiques (comptage, assurance, classement, etc.).
- On peut suivre plusieurs métriques simultanément : une pour XGBoost, d'autres pour les décideurs.
- Les fonctions objectives et métriques personnalisées permettent d'intégrer des contraintes métier (variance, asymétrie, etc.), au prix de calculer gradient et hessien.
- La combinaison d'une objective custom et d'une perte standard via un mélange est souvent nécessaire pour préserver la qualité prédictive tout en respectant la contrainte.
