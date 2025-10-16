# Parcours TP8 — Prétraitement et sélection des variables

[⬅️ Retour au sommaire](../../LISEZMOI.md)

Ce parcours est désormais scindé en **quatre TP complémentaires**, chacun dédié à une étape clé du prétraitement. Ils peuvent être réalisés indépendamment, mais il est recommandé de les suivre dans l'ordre afin de construire progressivement un pipeline robuste.

| TP | Thématique principale | Compétences travaillées |
| --- | --- | --- |
| [TP8.1 — Sélection de variables explicatives](TP8_1_Selection_Variables.md) | Comprendre et comparer différentes techniques de sélection de variables. | Analyse statistique, validation croisée, interprétation de modèles. |
| [TP8.2 — Normalisation et encodage](TP8_2_Normalisation_Encodage.md) | Mettre en place des pipelines de transformation adaptés aux types de variables. | Choix des transformateurs, gestion des fuites de données, comparaison de modèles. |
| [TP8.3 — Détection et traitement des valeurs aberrantes](TP8_3_Valeurs_Aberrantes.md) | Identifier et traiter les observations aberrantes avant l'entraînement. | Méthodes statistiques, détection automatique, robustesse des modèles. |
| [TP8.4 — Classification déséquilibrée](TP8_4_Classification_Desequilibree.md) | Adapter les modèles de classification aux jeux de données déséquilibrés. | Stratégies de rééchantillonnage, métriques adaptées, paramétrage d'algorithmes. |

## Objectifs pédagogiques transverses

- Comprendre les étapes clés de préparation des variables avant l'entraînement d'un modèle.
- Savoir sélectionner les variables pertinentes et réduire la dimension.
- Mettre en œuvre la normalisation des variables quantitatives et l'encodage des variables qualitatives.
- Identifier et traiter les observations aberrantes (outliers).
- Adapter les pipelines de classification aux jeux de données déséquilibrés.
- Discuter des impacts spécifiques sur des algorithmes comme XGBoost.

## Prérequis

- Python, `pandas`, `numpy`, `matplotlib`, `scikit-learn`.
- Connaissances de base sur la validation croisée, les pipelines et l'évaluation de modèles (accuracy, f1-score, ROC/AUC, etc.).

## Dataset recommandé

L'ensemble des TP s'appuie sur le dataset **Adult Income** (UCI Census Income) accessible via `fetch_openml`. L'objectif est de prédire si le revenu annuel dépasse 50 k$ à partir de variables socio-démographiques (numériques et catégorielles) avec un déséquilibre modéré.

```python
from sklearn.datasets import fetch_openml
import pandas as pd

adult = fetch_openml(name="adult", version=2, as_frame=True)
X = adult.data
y = adult.target
```

Vous pouvez toutefois utiliser un autre dataset tabulaire si vous le justifiez dans vos livrables.

## Livrables attendus

Pour chaque TP, fournissez :

1. Un notebook Jupyter documenté contenant le code, les visualisations et les réponses aux questions.
2. Une synthèse de 5 à 10 lignes mettant en avant les principaux enseignements (choix effectués, limites rencontrées, axes d'amélioration).
3. Les hyperparamètres retenus et les métriques pertinentes pour la comparaison de vos modèles.

Une synthèse globale (10 à 15 lignes) est demandée en fin de parcours pour relier vos conclusions.
