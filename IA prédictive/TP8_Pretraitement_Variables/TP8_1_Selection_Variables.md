# TP8.1 — Sélection de variables explicatives

[⬅️ Retour au sommaire](../../LISEZMOI.md)

## Positionnement pédagogique
- **Durée indicative** : 3 heures.
- **Niveau** : étudiants de 3e année de BUT Informatique (parcours IA/Data).
- **Compétences visées** : compréhension des critères de sélection, mise en œuvre de pipelines scikit-learn, analyse critique des résultats.

## Objectifs d'apprentissage
1. Identifier les variables pertinentes pour prédire une cible binaire.
2. Comparer plusieurs approches de sélection et comprendre leurs hypothèses.
3. Mesurer l'impact de la sélection sur les performances et la complexité des modèles.

## Mise en route
1. Charger le dataset Adult Income (`fetch_openml`).
2. Séparer les features (`X`) de la cible (`y`).
3. Identifier les colonnes numériques et catégorielles (`select_dtypes`).
4. Construire un `ColumnTransformer` minimal pour gérer les types de variables :
   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder

   numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
   categorical_features = X.select_dtypes(include=["object", "category"]).columns

   preprocess = ColumnTransformer([
       ("num", "passthrough", numeric_features),
       ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
   ])
   ```

> 💡 *Astuce pédagogique* : Vérifiez la taille des données transformées (`shape`) pour prendre conscience de l'explosion dimensionnelle liée au one-hot encoding.

## Activité 1 — Analyse univariée (45 min)
1. **Objectif** : mesurer l'association de chaque variable indépendante avec la variable cible.
2. **Méthodes proposées** :
   - `SelectKBest` avec `chi2` (sur données catégorielles encodées et positives) ;
   - `SelectKBest` avec `mutual_info_classif` (robuste aux relations non linéaires).
3. **Plan d'action** :
   - Intégrer le `SelectKBest` dans un pipeline avec une régression logistique.
   - Tester `k ∈ {10, 20, 40}`.
   - Utiliser une validation croisée (`cross_val_score`, `StratifiedKFold`) pour estimer `accuracy`, `f1` et `roc_auc`.
4. **Analyse attendue** :
   - Graphique `k` vs `score` pour chaque métrique.
   - Identification des variables sélectionnées pour chaque `k` (lister les noms grâce à `get_support`).
   - Commentaire : quels types de variables ressortent ? Y a-t-il des surprises ?

> 🧪 *À rendre* : tableau comparatif des scores, interprétation écrite (5 lignes).

## Activité 2 — Sélection basée sur un modèle (60 min)
1. **Objectif** : utiliser un modèle capable de pondérer l'importance des variables pour décider lesquelles conserver.
2. **Modèles suggérés** :
   - `RandomForestClassifier` avec `SelectFromModel` ;
   - `LogisticRegression(penalty="l1", solver="liblinear")` pour appliquer une pénalité Lasso.
3. **Étapes détaillées** :
   - Créer deux pipelines séparés (forêt aléatoire vs régression logistique pénalisée).
   - Ajuster l'hyperparamètre de régularisation (`C`) ou le seuil d'importance pour contrôler le nombre de variables retenues.
   - Évaluer chaque pipeline via validation croisée (métriques : `f1`, `balanced_accuracy`, `roc_auc`).
4. **Questions de réflexion** :
   - Les variables sélectionnées par la forêt sont-elles identiques à celles du Lasso ? Pourquoi ?
   - Comment justifier le choix d'un seuil (par exemple `median`) pour `SelectFromModel` ?

> 📌 *Aller plus loin* : visualiser les importances normalisées (`feature_importances_` ou `coef_`) et discuter de leur interprétation.

## Activité 3 — Sélection séquentielle (45 min)
1. **Objectif** : comprendre les approches incrémentales (`SequentialFeatureSelector`).
2. **Méthodologie** :
   - Utiliser une régression logistique comme estimateur de base.
   - Configurer deux sélecteurs : `direction="forward"` et `direction="backward"`.
   - Limiter le nombre de variables recherchées (ex. `n_features_to_select=25`) pour contenir les temps de calcul.
3. **Consignes** :
   - Mesurer le temps d'exécution (module `time`).
   - Comparer les scores de validation croisée aux méthodes précédentes.
   - Discuter des compromis entre performance et coût de calcul.

> ⏱️ *Attention* : documentez les temps d'exécution dans le notebook pour argumenter vos choix.

## Activité 4 — Impact sur XGBoost (30 min)
1. **Contexte** : XGBoost gère déjà un mécanisme de sélection interne via le gain d'information.
2. **Expérimentation** :
   - Créer un pipeline avec prétraitement minimal (`OneHotEncoder` uniquement pour les catégories).
   - Entrainer deux modèles XGBoost :
     - `XGBClassifier` sur l'ensemble complet ;
     - le même modèle sur les variables retenues par la méthode jugée la plus pertinente (Activités 1 à 3).
   - Comparer temps d'entraînement, `f1`, `roc_auc`.
3. **Analyse** : commenter l'intérêt (ou non) de filtrer les variables avant XGBoost.

## Synthèse à produire
- Tableau récapitulatif : méthode de sélection / nombre de variables / métriques principales / temps d'entraînement.
- Conclusion (8 à 10 lignes) :
  - Quelle méthode est la plus adaptée à ce dataset ?
  - Quelles sont les limites observées ?
  - Quels critères utiliseriez-vous pour choisir une méthode dans un nouveau projet ?

## Ressources complémentaires
- Documentation scikit-learn : [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- Article de blog recommandé : *Feature Selection Techniques in Machine Learning* (Analytics Vidhya).
- Tutoriel vidéo (facultatif) : *Feature Selection in Python* (StatQuest).
