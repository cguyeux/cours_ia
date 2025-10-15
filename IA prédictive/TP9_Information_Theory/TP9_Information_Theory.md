# Théorie de l'information et analyse de données tabulaires

Dans ce notebook, nous allons introduire progressivement des notions fondamentales de théorie de l'information et montrer comment elles peuvent aider à analyser des données tabulaires. Nous aborderons notamment :

- **Entropie** : mesure de l'incertitude d'une variable aléatoire (interprétation en nombre moyen de questions oui/non).
- (Option) **Compression sans perte** : un exemple de code binaire optimal illustrant le lien entre entropie et longueur moyenne minimale d'un code/décision.
- **Entropie conditionnelle et information mutuelle** \(H(Y), H(Y \mid X), I(Y;X)\) avec estimation empirique par discrétisation des données.
- **Application à un jeu de données synthétique** ("arrivées aux urgences") : variables explicatives (jour, température, épidémie de grippe...) et variable cible (comptage de patients).
- **Modélisation prédictive avec XGBoost (objectif Poisson)** : construction d'un modèle de régression pour prédire le nombre de patients, comparaison de modèles à faible vs forte profondeur.
- **Interprétation des résultats** : lien entre entropie et profondeur d'arbre (réduction d'incertitude au fil des splits), rôle informatif de chaque variable via l'information mutuelle et mesure de la réduction d'incertitude (lien avec la divergence de Kullback-Leibler).

Chaque section comprend des explications pédagogiques et du code Python exécutable illustrant les concepts (calculs d'entropie, construction d'arbres, entraînement XGBoost...). Des visualisations et sorties chiffrées aideront à interpréter les résultats. Enfin, un TP "À vous de jouer" vous proposera des manipulations pour explorer les effets de ces notions sur d'autres jeux de données ou paramètres.

## 1. Entropie : mesurer l'incertitude d'une variable aléatoire

**Définition.** L'entropie \(H(X)\) d'une variable aléatoire discrète \(X\) mesure l'incertitude moyenne associée à \(X\). Mathématiquement, pour une distribution prenant des valeurs \(x_i\) avec probabilités \(p_i = P(X = x_i)\), l'entropie se définit par :

\[
H(X) = - \sum_i p_i \log_2(p_i).
\]

L'unité est le **bit** si l'on utilise le logarithme en base 2. Intuitivement, l'entropie correspond au nombre moyen de questions oui/non qu'un observateur doit poser pour deviner la valeur de \(X\). Plus \(X\) est imprévisible (distribution uniforme par exemple), plus son entropie est élevée. À l'inverse, si \(X\) prend toujours la même valeur, son entropie est nulle (aucune incertitude).

### Exemples simples

- Une pièce équilibrée (pile ou face équiprobables) a une entropie de **1 bit**, car une question oui/non suffit en moyenne à deviner l'issue.
- Une pièce biaisée (qui tombe face 70 % du temps) a une entropie plus faible : l'incertitude est partiellement réduite car une issue est favorisée.
- Un dé à 6 faces équilibré a une entropie plus élevée (~2.585 bits), car il y a 6 issues équiprobables.
- Un dé biaisé (40 % une face, 20 % une autre, le reste à 10 %) présente une entropie intermédiaire.

Calculons ces entropies avec Python.


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

Pour \(n\) issues possibles, l'entropie maximale est \(\log_2(n)\) bits, atteinte lorsque les \(n\) issues sont équiprobables. L'entropie minimale est 0 bit, lorsque l'une des issues a probabilité 1 (distribution dégénérée).

### Un mot sur la compression

L'entropie fournit un plancher théorique pour la longueur moyenne d'un code binaire optimal (Shannon). Les algorithmes pratiques comme Huffman approchent ce plancher, mais nous n'entrerons pas dans les détails ici et nous concentrerons sur l'usage des entropies pour l'analyse de données tabulaires.
