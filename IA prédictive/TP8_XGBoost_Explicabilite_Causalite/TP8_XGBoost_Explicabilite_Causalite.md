# TP 8 : XGBoost – Explicabilité et Causalité

[⬅️ Retour au README](../../README.md)

## Objectifs pédagogiques

- Comprendre les différents indicateurs d'importance de variables fournis par XGBoost et savoir les visualiser.
- Interpréter le sens des mesures `weight`, `gain` et `cover` et construire un classement pondéré des variables.
- Expliquer une prédiction individuelle à l'aide des valeurs de Shapley (SHAP) et communiquer les résultats à l'écrit.
- Manipuler une bibliothèque d'inférence causale et relier corrélations observées et effets causaux estimés.
- Mettre en pratique ces notions sur des exemples concrets et proposer ses propres analyses.

## 1. Importance des variables avec `feature_importances_`

Dans la plupart des API de haut niveau (scikit-learn, `xgboost.sklearn`), les modèles XGBoost exposent un attribut `feature_importances_`. Celui-ci contient, pour chaque variable, la **proportion du gain total** accumulé par les arbres qui utilisent cette variable lors des splits. Plus précisément :

1. Lors de chaque itération, XGBoost ajoute un arbre qui réduit la fonction de perte (`objective`).
2. Chaque split dans l'arbre est évalué en fonction du **gain de perte** qu'il procure : la réduction de la perte totale grâce à ce split.
3. Le gain obtenu est crédité à la variable utilisée. À la fin de l'entraînement, on additionne et normalise ces gains par variable.

### 1.1 Exemple pratique

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Charger un jeu de données supervisé
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entraîner un modèle XGBoost de type sklearn
modele = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
modele.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

# Récupérer l'importance des variables
importances = modele.feature_importances_
```

### 1.2 Affichage tabulaire et graphique

```python
import pandas as pd

fi = (
    pd.Series(importances, index=X.columns)
    .sort_values(ascending=False)
    .rename("gain_normalise")
)
print(fi.head(10))
```

Pour un graphique prêt à l'emploi, utilisez la fonction utilitaire d'XGBoost :

```python
xgb.plot_importance(modele, importance_type="gain", max_num_features=15)
```

Le paramètre `importance_type` contrôle la métrique affichée (`gain` par défaut). La figure montre la contribution relative de chaque variable à la réduction de la perte totale. Les barres sont normalisées pour sommer à 1.

### 1.3 Interprétation

- Les variables avec un gain élevé sont celles qui, collectivement, ont le plus aidé à réduire la perte. Elles sont souvent de bons candidats pour l'analyse.
- Une importance proche de zéro indique que la variable a été très peu, voire jamais, utilisée dans les splits.
- Attention : cette mesure reflète l'**importance globale** moyenne sur l'entraînement. Elle ne capture pas l'effet sur des observations individuelles ni les interactions complexes entre variables.

## 2. Comprendre `weight`, `gain` et `cover`

Lorsque l'on accède directement à l'objet `Booster` (via `modele.get_booster()`), XGBoost fournit trois indicateurs complémentaires :

- **`weight`** : nombre de fois où la variable apparaît dans un split (tous arbres confondus). Indique la fréquence d'utilisation.
- **`gain`** : gain moyen de perte apporté par les splits utilisant cette variable. Reflète la qualité des splits.
- **`cover`** : nombre moyen d'échantillons (ou poids) concernés par les splits utilisant cette variable. Mesure la « couverture » des données.

### 2.1 Récupération des trois indicateurs

```python
booster = modele.get_booster()
importance_weight = booster.get_score(importance_type="weight")
importance_gain = booster.get_score(importance_type="gain")
importance_cover = booster.get_score(importance_type="cover")
```

Les dictionnaires retournés ont pour clés les noms internes des variables (`f0`, `f1`, ...). Pour les relier aux noms d'origine :

```python
feature_names = booster.feature_names
mapping = {f"f{i}": name for i, name in enumerate(feature_names)}
```

### 2.2 Construction d'un tableau synthétique

```python
import pandas as pd

importance_df = (
    pd.DataFrame({
        "feature": feature_names,
        "weight": [importance_weight.get(f"f{i}", 0) for i in range(len(feature_names))],
        "gain": [importance_gain.get(f"f{i}", 0.0) for i in range(len(feature_names))],
        "cover": [importance_cover.get(f"f{i}", 0.0) for i in range(len(feature_names))],
    })
    .assign(
        gain_total=lambda df: df["gain"] * df["weight"],
        cover_total=lambda df: df["cover"] * df["weight"],
    )
)
```

Dans cet exemple :

- `gain_total` = somme des gains pour tous les splits (gain moyen × nombre de splits).
- `cover_total` = nombre total d'exemples couverts.

### 2.3 Classement pondéré personnalisé

Pour obtenir un score composite, on peut combiner ces indicateurs. Par exemple :

```python
from sklearn.preprocessing import minmax_scale

importance_df = importance_df.assign(
    score_combine=lambda df: (
        0.5 * minmax_scale(df["gain_total"]) +
        0.3 * minmax_scale(df["cover_total"]) +
        0.2 * minmax_scale(df["weight"])
    )
)

classement = importance_df.sort_values("score_combine", ascending=False)
print(classement.head(10))
```

Cette moyenne pondérée privilégie les variables qui génèrent un fort gain total, couvrent beaucoup d'observations et sont régulièrement utilisées. Les pondérations (0.5 / 0.3 / 0.2) sont à ajuster selon les priorités métier.

### 2.4 Limites et bonnes pratiques

- `weight` favorise les variables utilisées dans de nombreux splits, même si les gains sont faibles.
- `gain` peut être dominé par quelques splits très profitables mais rares.
- `cover` met en avant les variables qui impactent de larges segments des données.
- Croiser ces trois points de vue permet de mieux comprendre le rôle global des variables dans le modèle.

## 3. Explications locales avec les valeurs SHAP

Les **SHAP values** (SHapley Additive exPlanations) s'appuient sur la théorie des jeux coopératifs. Pour une observation donnée :

- On considère chaque variable comme un « joueur » qui contribue à la prédiction.
- La valeur SHAP mesure la contribution moyenne marginale de cette variable en comparant tous les sous-ensembles possibles de variables.
- La somme des valeurs SHAP + la valeur de base (baseline) = prédiction du modèle.

### 3.1 Calculer les valeurs SHAP pour XGBoost

Installez au préalable la bibliothèque :

```bash
pip install shap
```

```python
import shap
import numpy as np

explainer = shap.TreeExplainer(modele)  # fonctionne pour les modèles basés sur des arbres
shap_values = explainer.shap_values(X_valid)
base_value = explainer.expected_value
baseline_proba = 1 / (1 + np.exp(-base_value))
```

Pour une classification binaire, `shap_values` est un tableau de la même taille que `X_valid` (n_observations × n_features). Chaque ligne donne la contribution signée de chaque variable à la prédiction en log-odds (sortie brute). `base_value` correspond à la moyenne des sorties du modèle sur le jeu d'entraînement.

### 3.2 Exemple détaillé d'une prédiction

```python
import numpy as np

idx = 0  # choisir une observation de validation
x = X_valid.iloc[idx]
y_pred_proba = modele.predict_proba(X_valid.iloc[[idx]])[0, 1]
y_pred_logodds = modele.predict(X_valid.iloc[[idx]], output_margin=True)[0]

contributions = pd.Series(shap_values[idx], index=X.columns)
print("Valeur de base (log-odds):", base_value)
print("Probabilité moyenne associée:", baseline_proba)
print("Somme contributions:", contributions.sum())
print("Log-odds prédit:", y_pred_logodds)
print("Probabilité prédite (après sigmoïde):", y_pred_proba)
```

La sortie montre que :

- `base_value` est la prédiction moyenne en log-odds.
- `contributions.sum()` est exactement égale au log-odds prédit.
- Chaque valeur SHAP indique de combien la variable décale la prédiction par rapport à la base (positif = augmente la proba, négatif = la réduit).

### 3.3 Visualiser et lire les graphiques SHAP

1. **Waterfall plot** (explication d'une observation) :

    ```python
    shap_explication = shap.Explanation(
        values=shap_values[idx],
        base_values=base_value,
        data=x,
        feature_names=X.columns,
    )
    shap.plots.waterfall(shap_explication)
    ```

    À lire de gauche à droite : la base, puis les contributions positives (rouge) et négatives (bleu) jusqu'à la sortie.

2. **Summary plot** (vue globale) :

    ```python
    shap.summary_plot(shap_values, X_valid)
    ```

    - L'axe horizontal indique l'impact (SHAP value).
    - La couleur encode la valeur de la variable (rouge = élevée, bleu = faible).
    - Les points les plus à droite sont ceux qui augmentent fortement la prédiction.

3. **Dependence plot** (interaction variable) :

    ```python
    shap.dependence_plot("mean radius", shap_values, X_valid)
    ```

    Il montre comment la contribution d'une variable évolue en fonction de sa valeur et, en couleur, l'une des variables les plus corrélées (interaction).

### 3.4 Générer une explication textuelle

Une fois les contributions calculées, vous pouvez produire un résumé en langage naturel, par exemple en utilisant un prompt destiné à un modèle de langage :

```
Contexte : modèle XGBoost de classification du cancer du sein.
Observation : {valeurs des variables clés}
Baseline (log-odds) : {base_value:.3f} => probabilité moyenne {baseline_proba:.2%}.
Contributions principales :
- {feature_1} : +{shap_1:.3f} (valeur observée {x1}), augmente la probabilité car ...
- {feature_2} : -{shap_2:.3f} (valeur observée {x2}), réduit la probabilité car ...
Prédiction finale : log-odds {logodds:.3f} => probabilité {proba:.2%}.
Explique en 5 phrases simples à un public non spécialiste.
```

Il suffit d'alimenter les espaces `{...}` avec les résultats de l'analyse pour obtenir un texte cohérent. Cette approche encourage les étudiantes et étudiants à traduire des valeurs numériques en arguments compréhensibles.

> 💡 `baseline_proba` peut être calculé via `baseline_proba = 1 / (1 + np.exp(-base_value))`, ce qui correspond à la probabilité moyenne prédite par le modèle.

## 4. Introduction à la causalité avec DoWhy

La corrélation mesurée par les importances ou SHAP ne garantit pas un **lien causal**. Pour raisonner sur des scénarios contrefactuels (« que se passerait-il si... »), on s'appuie sur l'inférence causale.

[DoWhy](https://github.com/py-why/dowhy) est une bibliothèque Python moderne qui propose un cadre en quatre étapes : modéliser, identifier, estimer et réfuter.

### 4.1 Installation

```bash
pip install dowhy econml graphviz pygraphviz
```

### 4.2 Exemple concret : impact d'une campagne marketing

Supposons que l'on dispose d'un jeu de données avec :

- `traitement` : indicateur (0/1) que la cliente a reçu une campagne marketing.
- `revenu` : revenu annuel de la cliente.
- `anciennete` : ancienneté dans le programme fidélité.
- `depense` : montant dépensé après la campagne (variable cible).

Nous cherchons l'effet causal de `traitement` sur `depense`, en contrôlant les variables de confusion (`revenu`, `anciennete`).

```python
import pandas as pd
import numpy as np
from dowhy import CausalModel

n = 2000
rng = np.random.default_rng(0)
revenu = rng.normal(45000, 10000, size=n)
anciennete = rng.integers(1, 8, size=n)
# Les clientes avec un revenu élevé reçoivent plus souvent la campagne
propension = 1 / (1 + np.exp(-0.00008 * (revenu - 42000) + 0.3 * (anciennete - 3)))
traitement = rng.binomial(1, propension)
# Effet causal réel : +80 € en moyenne
bruit = rng.normal(0, 150, size=n)
depense = 500 + 0.05 * revenu + 15 * anciennete + 80 * traitement + bruit

df = pd.DataFrame({
    "revenu": revenu,
    "anciennete": anciennete,
    "traitement": traitement,
    "depense": depense,
})
```

### 4.3 Spécifier le graphe causal

```python
modele_causal = CausalModel(
    data=df,
    treatment="traitement",
    outcome="depense",
    graph="""
        digraph {
            revenu -> traitement;
            revenu -> depense;
            anciennete -> traitement;
            anciennete -> depense;
            traitement -> depense;
        }
    """
)
modele_causal.view_model(layout="dot")  # nécessite graphviz
```

Le graphe encode les relations causales supposées. Les arcs partent des causes vers les effets.

### 4.4 Identification et estimation

```python
identification = modele_causal.identify_effect()
print(identification)

estimateur = modele_causal.estimate_effect(
    identification,
    method_name="backdoor.propensity_score_matching",
)
print("Effet moyen du traitement (ATE) estimé:", estimateur.value)
```

Ici, DoWhy applique un ajustement par score de propension (`propensity_score_matching`). Si le modèle est correctement spécifié, l'ATE estimé doit se rapprocher de la valeur réelle (≈ 80 €).

### 4.5 Tests de réfutation

DoWhy propose des tests pour vérifier la robustesse de l'estimation :

```python
refutation = modele_causal.refute_estimate(
    identification,
    estimateur,
    method_name="placebo_treatment_refuter",
)
print(refutation)
```

- **Placebo treatment** : remplace le traitement par une variable aléatoire ; l'effet doit alors être proche de 0.
- D'autres méthodes (`data_subset_refuter`, `add_unobserved_common_cause`) permettent de tester la sensibilité aux hypothèses.

### 4.6 Lien avec l'explicabilité

- Les importances et SHAP nous informent sur **comment** le modèle se sert des variables pour prédire.
- L'analyse causale répond à **ce qui se passerait** si nous intervenions sur une variable (par exemple envoyer ou non une campagne).
- Combiner les deux approches aide à prioriser les actions : une variable peut être importante pour la prédiction mais avoir un effet causal faible (corrélation due à un facteur commun).

## Exercices

1. **Importances multiples** : Entraînez un `XGBRegressor` sur le jeu de données `CaliforniaHousing`. Comparez les importances `gain`, `weight` et `cover`. Quels groupes de variables ressortent selon chaque métrique ?
2. **Score personnalisé** : Proposez votre propre combinaison pondérée des trois indicateurs d'importance. Justifiez vos choix et identifiez les variables prioritaires pour un audit.
3. **SHAP détaillé** : Choisissez deux observations (une prédiction correcte, une incorrecte) et réalisez une explication SHAP complète : tableau de contributions, waterfall plot, résumé textuel.
4. **Sensibilité aux interactions** : Utilisez `shap.dependence_plot` pour explorer l'interaction entre deux variables. Comment interpréter les variations de contributions ?
5. **Expérience causale** : Reprenez l'exemple DoWhy et modifiez le générateur de données pour introduire une variable omise (non observée) influençant à la fois le traitement et l'issue. Observez l'impact sur l'estimation de l'ATE et discutez de la robustesse des tests de réfutation.
6. **Étude de cas** : Proposez un cas métier (ex : recommandation produit) et décrivez comment vous utiliseriez XGBoost pour prédire, SHAP pour expliquer, puis DoWhy (ou autre bibliothèque causale) pour estimer l'effet d'une intervention. Listez les données nécessaires et les hypothèses à vérifier.

Bon TP !
