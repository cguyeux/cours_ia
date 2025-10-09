# TP7.4 — Classification déséquilibrée

[⬅️ Retour au README](../../README.md)

## Positionnement pédagogique
- **Durée indicative** : 3 heures.
- **Niveau** : BUT3 Informatique.
- **Compétences visées** : traitement des jeux de données déséquilibrés, choix de métriques adaptées, paramétrage d'algorithmes (scikit-learn et XGBoost).

## Objectifs d'apprentissage
1. Diagnostiquer le niveau de déséquilibre et ses impacts sur les métriques habituelles.
2. Mettre en place des techniques de rééchantillonnage et de pondération.
3. Ajuster le seuil de décision et interpréter les courbes ROC/PR.
4. Paramétrer XGBoost pour des classes minoritaires.

## Préparation
1. Recharger les datasets `X_train`, `X_test`, `y_train`, `y_test` préparés précédemment (stratifiés).
2. Réutiliser le pipeline de prétraitement validé dans TP7.2 (normalisation + encodage) comme bloc de base.
3. Installer `imblearn` si besoin (`pip install imbalanced-learn`).

## Activité 1 — Diagnostic et métriques (25 min)
1. Calculer la proportion de chaque classe dans le train et le test.
2. Entraîner une régression logistique *sans* rééquilibrage et évaluer : `accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy`, `roc_auc`, `average_precision`.
3. Commenter pourquoi l'accuracy peut être trompeuse dans ce contexte.
4. Visualiser la matrice de confusion et identifier les coûts métiers associés aux faux positifs/faux négatifs.

## Activité 2 — Rééchantillonnage (60 min)
1. Tester trois stratégies dans une `ImbPipeline` (imblearn) :
   - `RandomOverSampler`
   - `SMOTE`
   - `RandomUnderSampler`
2. Pour chaque stratégie, utiliser deux modèles : régression logistique et `RandomForestClassifier` (avec `class_weight=None`).
3. Utiliser une validation croisée stratifiée (5 folds) et comparer `f1`, `recall`, `balanced_accuracy`.
4. Discuter :
   - Comment le sur-échantillonnage affecte-t-il le temps d'entraînement ?
   - Quelles méthodes semblent le mieux préserver la diversité des données ?

> 📈 *Visualisation suggérée* : courbes Precision-Recall pour chaque stratégie.

## Activité 3 — Pondération des classes (35 min)
1. Évaluer l'effet de `class_weight="balanced"` pour :
   - Régression logistique (`LogisticRegression`)
   - SVM linéaire (`LinearSVC`)
2. Comparer les résultats à ceux de l'Activité 2.
3. Introduire la notion de `scale_pos_weight` dans XGBoost :
   ```python
   from xgboost import XGBClassifier

   scale_pos_weight = len(y_train[y_train == "<=50K"]) / len(y_train[y_train == ">50K"])
   model = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr")
   ```
4. Tester XGBoost avec et sans pondération.

## Activité 4 — Optimisation du seuil de décision (35 min)
1. À partir du meilleur modèle (Activité 2 ou 3), récupérer les probabilités (`predict_proba`).
2. Tracer la courbe Precision-Recall (`sklearn.metrics.precision_recall_curve`).
3. Définir un seuil maximisant le `f1` ou minimisant un coût métier défini par les étudiants.
4. Implémenter une fonction utilitaire qui, pour un seuil donné, renvoie la matrice de confusion, `precision`, `recall`, `f1`.
5. Illustrer l'effet de trois seuils distincts (ex. 0.3, 0.5, 0.7) sur les métriques.

## Activité 5 — Synthèse XGBoost vs modèles linéaires (25 min)
1. Comparer deux pipelines :
   - Prétraitement complet + `LogisticRegression` avec la meilleure stratégie (rééchantillonnage ou pondération) + seuil optimisé.
   - Prétraitement minimal + `XGBClassifier` avec `scale_pos_weight` et early stopping (utiliser `eval_set`).
2. Évaluer sur le jeu de test : `f1`, `recall`, `roc_auc`, `average_precision`, temps d'entraînement.
3. Discuter des avantages/inconvénients :
   - Interprétabilité vs performance brute.
   - Sensibilité aux hyperparamètres.
   - Facilité de déploiement.

## Synthèse attendue
- Tableau comparatif des stratégies testées (rééchantillonnage, pondération, seuils) avec les métriques clés.
- Recommandations (8 à 10 lignes) :
  - Quelle stratégie adopter pour un contexte où les faux négatifs sont coûteux ?
  - Comment monitorer les performances en production (drift de classe) ?
  - Quels indicateurs suivre en continu (precision@k, recall@k, etc.) ?

## Ressources
- Documentation imbalanced-learn : [https://imbalanced-learn.org/stable/](https://imbalanced-learn.org/stable/)
- Documentation scikit-learn : [Imbalanced datasets](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- Article : *Dealing with Imbalanced Data in Machine Learning* (Analytics Vidhya).
