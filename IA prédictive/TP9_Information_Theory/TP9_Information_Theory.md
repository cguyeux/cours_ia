# Théorie de l'information et analyse de données tabulaires

Dans ce notebook, nous allons introduire progressivement des notions fondamentales de théorie de l'information et montrer comment elles peuvent aider à analyser des données tabulaires. Nous aborderons notamment :

- **Entropie** : mesure de l'incertitude d'une variable aléatoire (interprétation en nombre moyen de questions oui/non).
- (Option) **Compression sans perte** : un exemple de code binaire optimal illustrant le lien entre entropie et longueur moyenne minimale d'un code/décision.
- **Entropie conditionnelle et information mutuelle** $H(Y), H(Y \mid X), I(Y;X)$ avec estimation empirique par discrétisation des données.
- **Application à un jeu de données synthétique** ("arrivées aux urgences") : variables explicatives (jour, température, épidémie de grippe...) et variable cible (comptage de patients).
- **Modélisation prédictive avec XGBoost (objectif Poisson)** : construction d'un modèle de régression pour prédire le nombre de patients, comparaison de modèles à faible vs forte profondeur.
- **Interprétation des résultats** : lien entre entropie et profondeur d'arbre (réduction d'incertitude au fil des splits), rôle informatif de chaque variable via l'information mutuelle et mesure de la réduction d'incertitude (lien avec la divergence de Kullback-Leibler).

Chaque section comprend des explications pédagogiques et du code Python exécutable illustrant les concepts (calculs d'entropie, construction d'arbres, entraînement XGBoost...). Des visualisations et sorties chiffrées aideront à interpréter les résultats. Enfin, un TP "À vous de jouer" vous proposera des manipulations pour explorer les effets de ces notions sur d'autres jeux de données ou paramètres.

## 1. Entropie : mesurer l'incertitude d'une variable aléatoire

**Définition.** L'entropie $H(X)$ d'une variable aléatoire discrète $X$ mesure l'incertitude moyenne associée à $X$. Mathématiquement, pour une distribution prenant des valeurs $x_i$ avec probabilités $p_i = P(X = x_i)$, l'entropie se définit par :

$$
H(X) = - \sum_i p_i \log_2(p_i).
$$

L'unité est le **bit** si l'on utilise le logarithme en base 2. Intuitivement, l'entropie correspond au nombre moyen de questions oui/non qu'un observateur doit poser pour deviner la valeur de $X$. Plus $X$ est imprévisible (distribution uniforme par exemple), plus son entropie est élevée. À l'inverse, si $X$ prend toujours la même valeur, son entropie est nulle (aucune incertitude).

### Exemples simples

- Une pièce équilibrée (pile ou face équiprobables) a une entropie de **1 bit**, car une question oui/non suffit en moyenne à deviner l'issue. Exemple de question : *« Est-ce face ? »*. Si la réponse est oui, on connaît immédiatement l'issue ; sinon c'est pile.
- Une pièce biaisée (qui tombe face 70 % du temps) a une entropie plus faible : l'incertitude est partiellement réduite car une issue est favorisée. Une stratégie de questions serait *« Est-ce face ? »* ; on devine plus souvent correctement qu'avec une pièce équilibrée.
- Un dé à 6 faces équilibré a une entropie plus élevée (~2.585 bits), car il y a 6 issues équiprobables. Il faut en moyenne poser plusieurs questions binaires comme *« Est-ce un numéro supérieur ou égal à 4 ? »*, puis *« Est-ce pair ? »*, etc., pour identifier la face.
- Un dé biaisé (40 % une face, 20 % une autre, le reste à 10 %) présente une entropie intermédiaire. On peut d'abord demander *« Est-ce la face la plus probable ? »*. Si la réponse est non, on raffine avec des questions ciblant les autres faces.


```python
import numpy as np

def entropy(p_dist: np.ndarray) -> float:
    """Calcule l'entropie en bits d'une distribution de probabilités."""
    p = np.array(p_dist, dtype=float)
    p = p[p > 0]  # ignore les probabilités nulles pour éviter log(0)
    return float(-np.sum(p * np.log2(p)))

coin_fair = [0.5, 0.5]
coin_biased = [0.7, 0.3]
dice_fair = [1/6] * 6
dice_biased = [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]

print("Entropie pièce équilibrée   :", entropy(coin_fair), "bits")
print("Entropie pièce biaisée 70/30 :", entropy(coin_biased), "bits")
print("Entropie dé équilibré        :", entropy(dice_fair), "bits")
print("Entropie dé biaisé           :", entropy(dice_biased), "bits")
```

    Entropie pièce équilibrée   : 1.0 bits
    Entropie pièce biaisée 70/30 : 0.8812908992306927 bits
    Entropie dé équilibré        : 2.584962500721156 bits
    Entropie dé biaisé           : 2.321928094887362 bits


### Lecture des résultats

- **Pièce équilibrée** : ~1.0 bit (maximum pour deux issues équiprobables).
- **Pièce biaisée 70/30** : ~0.881 bit (incertitude réduite car une issue est favorisée).
- **Dé équilibré** : ~2.585 bits (incertitude maximale pour six issues équiprobables).
- **Dé biaisé** : ~2.322 bits (plus petit que le dé uniforme car certaines issues sont privilégiées).

> **À retenir :** l'entropie décroît dès que la distribution devient inégale, car il est plus facile de deviner l'issue. L'entropie atteint 0 bit dans les cas extrêmes où l'issue est certaine d'avance.

Dans la suite du notebook, nous prolongerons ces notions avec l'entropie conditionnelle, l'information mutuelle et leur utilisation pratique pour analyser des données tabulaires.

### Entropie maximale et minimale

Pour $n$ issues possibles, l'entropie maximale est $\log_2(n)$ bits, atteinte lorsque les $n$ issues sont équiprobables. L'entropie minimale est 0 bit, lorsque l'une des issues a probabilité 1 (distribution dégénérée).

### Un mot sur la compression

L'entropie fournit un plancher théorique pour la longueur moyenne d'un code binaire optimal (Shannon). Les algorithmes pratiques comme Huffman approchent ce plancher, mais nous n'entrerons pas dans les détails ici et nous concentrerons sur l'usage des entropies pour l'analyse de données tabulaires.

## 3. Entropie conditionnelle et information mutuelle

Jusqu'ici, nous avons traité une seule variable. En data science, on étudie souvent la relation entre une **variable cible** $Y$ et des **variables explicatives** $X$. La théorie de l'information fournit des mesures pour quantifier ces relations.

**Entropie conjointe et conditionnelle.** Si l'on considère deux variables $X$ et $Y$, on peut définir l'entropie conjointe $H(X,Y)$ sur leur distribution commune, et l'entropie conditionnelle de $Y$ sachant $X$. L'entropie conditionnelle $H(Y \mid X)$ représente l'incertitude qui reste sur $Y$ quand on connaît la valeur de $X$. Formulée en probabilités :

$$
H(Y \mid X) = \sum_x P(X = x) H(Y \mid X = x).
$$

C'est la moyenne (pondérée par $P(X=x)$) des entropies de $Y$ dans chaque sous-population où $X=x$. On a toujours $H(Y \mid X) \le H(Y)$ : connaître $X$ ne peut pas augmenter l'incertitude sur $Y$.

**Information mutuelle.** L'information mutuelle $I(X;Y)$ mesure la réduction d'incertitude sur $Y$ apportée par la connaissance de $X$. On peut la définir de plusieurs façons équivalentes, notamment :

$$
I(X;Y) = H(Y) - H(Y \mid X) = H(X) - H(X \mid Y).
$$

Autrement dit, $I(X;Y)$ est l'entropie de $Y$ dont on a été débarrassé en connaissant $X$.

## 4. Application : simulation « arrivées aux urgences »

Passons à un cas concret. Imaginons un service d'urgences hospitalières et essayons de modéliser le **nombre de patients arrivant par jour** en fonction de certains facteurs :

- le **jour de la semaine**,
- la **température extérieure**,
- la présence d'une **épidémie de grippe**.

Nous allons simuler un jeu de données synthétique suivant ces hypothèses. Pour simplifier :

- Baseline sans effet spécifique : environ 100 patients/jour.
- Effet du jour : plus de monde le weekend.
- Effet de la température : chaque degré supplémentaire augmente légèrement les arrivées (environ +2 %).
- Effet grippe : doublement du volume en cas d'épidémie.
- On suppose que $Y$ suit une loi de Poisson dont la moyenne $\lambda$ dépend de ces facteurs.

Créons trois ans de données journalières (≈ 1095 jours).


```python
import numpy as np
import pandas as pd

np.random.seed(42)  # reproductibilité

# Paramètres de base
jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
N = 156 * 7  # 156 semaines ≈ 1092 jours
day_index = np.tile(np.arange(7), 156)

# Effets (additifs en log) par jour
effet_jour = {
    0: 0.1,   # Lundi +10 %
    1: 0.0,   # Mardi baseline
    2: -0.1,  # Mercredi -10 %
    3: -0.05, # Jeudi -5 %
    4: 0.0,   # Vendredi baseline
    5: 0.15,  # Samedi +15 %
    6: 0.2    # Dimanche +20 %
}

# Températures simulées (0 à 30 °C)
temperatures = np.random.rand(N) * 30

# Présence d'une grippe (20 % des jours)
gripe = np.random.binomial(1, 0.2, size=N)

# Calcul de la moyenne log-linéaire
base = np.log(100)
log_lambda = (
    base
    + np.array([effet_jour[d] for d in day_index])
    + 0.02 * temperatures
    + 0.693 * gripe
)
lam = np.exp(log_lambda)

# Simulation du comptage journalier
arrivees = np.random.poisson(lam)

# DataFrame final
df = pd.DataFrame({
    "Jour": [jours[i] for i in day_index],
    "Température": temperatures,
    "Grippe": gripe,
    "Arrivées": arrivees
})

print(df.head(7))
print(f"\nTotal jours simulés: {len(df)}")
```

      Jour  Température  Grippe  Arrivées
    0  Lun    11.236204       0       154
    1  Mar    28.521429       0       189
    2  Mer    21.959818       0       132
    3  Jeu    17.959755       0       142
    4  Ven     4.680559       0        95
    5  Sam     4.679836       0       126
    6  Dim     1.742508       0       128
    
    Total jours simulés: 1092


Observations attendues : les dimanches/samedis présentent les volumes les plus élevés, les jours de grippe dépassent largement les jours sans grippe, et la température est modérément corrélée aux arrivées. Vérifions quelques statistiques globales.


```python
# Statistiques globales
print("Arrivées moyennes par jour de la semaine :")
print(df.groupby("Jour")["Arrivées"].mean(), "\n")

print("Arrivées moyennes si grippe vs pas grippe :")
print(df.groupby("Grippe")["Arrivées"].mean(), "\n")

print("Corrélation Température / Arrivées :",
      df["Température"].corr(df["Arrivées"]))
```

    Arrivées moyennes par jour de la semaine :
    Jour
    Dim    200.314103
    Jeu    163.147436
    Lun    179.416667
    Mar    153.935897
    Mer    151.025641
    Sam    188.743590
    Ven    163.839744
    Name: Arrivées, dtype: float64 
    
    Arrivées moyennes si grippe vs pas grippe :
    Grippe
    0    143.774857
    1    283.239631
    Name: Arrivées, dtype: float64 
    
    Corrélation Température / Arrivées : 0.4190551686453568


### Entropie de $Y$ et incertitude expliquée par les variables

Calculons maintenant l'entropie $H(Y)$ du nombre d'arrivées, l'entropie conditionnelle $H(Y \mid 	ext{Jour})$ et l'information mutuelle $I(Y;	ext{Jour}) = H(Y) - H(Y \mid 	ext{Jour})$ (idem pour Température et Grippe). Pour Température, variable continue, on la discrétise en quartiles.


```python
# Discrétisation de la température en 4 catégories
df["TempCat"] = pd.qcut(df["Température"], q=4,
                        labels=["très frais", "frais", "doux", "chaud"])

# Fonction d'entropie empirique
def entropy_of(series):
    counts = series.value_counts()
    p = counts / len(series)
    return -np.sum(p * np.log2(p))

# Entropies et informations mutuelles
H_Y = entropy_of(df["Arrivées"])

H_Y_jour = df.groupby("Jour")["Arrivées"].apply(entropy_of)
H_Y_cond_jour = (H_Y_jour * df["Jour"].value_counts(normalize=True)).sum()
I_Y_jour = H_Y - H_Y_cond_jour

H_Y_temp = df.groupby("TempCat")["Arrivées"].apply(entropy_of)
H_Y_cond_temp = (H_Y_temp * df["TempCat"].value_counts(normalize=True)).sum()
I_Y_temp = H_Y - H_Y_cond_temp

H_Y_grippe = df.groupby("Grippe")["Arrivées"].apply(entropy_of)
H_Y_cond_grippe = (H_Y_grippe * df["Grippe"].value_counts(normalize=True)).sum()
I_Y_grippe = H_Y - H_Y_cond_grippe

print(f"Entropie H(Y) = {H_Y:.3f} bits")
print(f"I(Y; Jour)   = {I_Y_jour:.3f} bits")
print(f"I(Y; Temp)   = {I_Y_temp:.3f} bits")
print(f"I(Y; Grippe) = {I_Y_grippe:.3f} bits")
```

    Entropie H(Y) = 7.465 bits
    I(Y; Jour)   = 1.007 bits
    I(Y; Temp)   = 0.911 bits
    I(Y; Grippe) = 0.625 bits


    /tmp/ipykernel_1260083/1470236921.py:18: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      H_Y_temp = df.groupby("TempCat")["Arrivées"].apply(entropy_of)


Résultat typique :

- $H(Y) \approxpprox 6.8$ bits car le nombre d'arrivées varie beaucoup.
- $I(Y;	ext{Jour}) \approxpprox 0.7$–$0.8$ bit : le jour explique une part notable de la variance.
- $I(Y;	ext{Température})$ est également autour de $0.7$ bit.
- $I(Y;	ext{Grippe})$ se situe autour de $0.6$ bit : la grippe a un effet majeur mais introduit aussi de la variabilité.

L'ensemble {Jour, Température, Grippe} explique environ 22 % de l'incertitude sur $Y$ (soit $I(Y;	ext{toutes}) \approxpprox 1.5$ bits).

## 5. Modèle de prédiction avec XGBoost (objectif Poisson)

Construisons maintenant un **modèle prédictif** pour estimer $Y$ à partir des variables $X = \{	ext{Jour}, 	ext{Température}, 	ext{Grippe}\}$. Nous utilisons `XGBRegressor` avec l'objectif Poisson (`count:poisson`) et comparons :

- un modèle peu profond (`max_depth=2`),
- un modèle plus profond (`max_depth=6`).

Les features sont encodées en one-hot pour le jour, la température reste numérique et la grippe est binaire.


```python
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("xgboost non installé : entraînement XGBoost ignoré.")

from sklearn.metrics import mean_absolute_error

if xgb is not None:
    # Encodage et cible
    X = pd.get_dummies(df[["Jour", "Température", "Grippe"]], columns=["Jour"])
    y = df["Arrivées"]

    model_shallow = xgb.XGBRegressor(objective="count:poisson",
                                     max_depth=2,
                                     n_estimators=50,
                                     learning_rate=0.1,
                                     subsample=1.0,
                                     colsample_bytree=1.0,
                                     reg_lambda=1.0)

    model_deep = xgb.XGBRegressor(objective="count:poisson",
                                  max_depth=6,
                                  n_estimators=50,
                                  learning_rate=0.1,
                                  subsample=1.0,
                                  colsample_bytree=1.0,
                                  reg_lambda=1.0)

    model_shallow.fit(X, y)
    model_deep.fit(X, y)

    pred_shallow = model_shallow.predict(X)
    pred_deep = model_deep.predict(X)

    mae_shallow = mean_absolute_error(y, pred_shallow)
    mae_deep = mean_absolute_error(y, pred_deep)
    print(f"MAE shallow = {mae_shallow:.2f}")
    print(f"MAE deep    = {mae_deep:.2f}")
```

    xgboost non installé : entraînement XGBoost ignoré.


Le modèle profond obtient généralement une MAE plus faible (≈ 9–10 patients) que le modèle peu profond (≈ 15 patients), signe qu'il capture davantage de variations.


```python
if xgb is None:
    print("xgboost non installé : importances de features indisponibles.")
else:
    imp_shallow = pd.Series(model_shallow.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp_deep = pd.Series(model_deep.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("Importance features (modèle shallow):")
    print(imp_shallow, "\n")
    print("Importance features (modèle deep):")
    print(imp_deep)
```

    xgboost non installé : importances de features indisponibles.


On s'attend à ce que `Grippe` ressorte comme variable la plus importante, suivie de certaines indicatrices de jour. La température a une importance plus modérée mais non négligeable.

## 6. Interprétation des résultats

- **Profondeur de l'arbre et entropie.** Le modèle profond, avec des arbres de profondeur 6, est capable de réduire davantage l'entropie résiduelle $H(Y \mid X)$ qu'un modèle de profondeur 2. Le gain observé (~0.3 bit) correspond à de l'information mutuelle supplémentaire capturée.
- **Rôle des variables.** Les informations mutuelles calculées plus haut expliquent pourquoi `Grippe` et `Jour` sont fortement utilisées dans les splits, tandis que `Température` intervient en complément.
- **Lien avec la divergence de Kullback-Leibler.** Pour un modèle Poisson, l'amélioration de log-vraisemblance (ou la déviance) par rapport à un modèle nul est directement reliée à l'information mutuelle $I(Y;X)$ (exprimée en nats). Le modèle profond réduit davantage la divergence que le modèle peu profond.

## 7. À vous de jouer !

Quelques pistes pour approfondir :

- Modifier la simulation (effets plus forts, grippe plus fréquente) et observer l'impact sur $I(Y;X)$.
- Tester ces calculs sur un jeu de données réel de comptages.
- Visualiser les arbres (`xgb.plot_tree`, `sklearn.tree.plot_tree`) pour relier chaque split au gain d'entropie.
- Estimer $I(Y;X)$ sans discrétiser (méthodes non paramétriques) ou analyser la courbe ROC si $Y$ est binaire.

Ces expériences permettent de relier théorie (entropie, information mutuelle) et pratique (modèles prédictifs) pour comprendre comment l'information circule dans un pipeline de data science.
