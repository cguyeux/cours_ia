# Théorie de l'information et analyse de données tabulaires

Dans ce notebook, nous allons introduire progressivement des notions fondamentales de théorie de l'information et montrer comment elles peuvent aider à analyser des données tabulaires. Nous aborderons notamment :

- **Entropie** : mesure de l'incertitude d'une variable aléatoire (interprétation en nombre moyen de questions oui/non).
- **Codage de Huffman** : un exemple de code binaire optimal illustrant le lien entre entropie et longueur moyenne minimale d'un code/décision.
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

- Une pièce équilibrée (pile ou face équiprobables) a une entropie de **1 bit**, car une question oui/non suffit en moyenne à deviner l'issue.
- Une pièce biaisée (qui tombe face 70 % du temps) a une entropie plus faible : l'incertitude est partiellement réduite car une issue est favorisée.
- Un dé à 6 faces équilibré a une entropie plus élevée (2.585 bits), car il y a 6 issues équiprobables.
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

- **Pièce équilibrée** : 1.0 bit (maximum pour deux issues équiprobables).
- **Pièce biaisée 70/30** : 0.881 bit (incertitude réduite car une issue est favorisée).
- **Dé équilibré** : 2.585 bits (incertitude maximale pour six issues équiprobables).
- **Dé biaisé** : 2.322 bits (plus petit que le dé uniforme car certaines issues sont privilégiées).

> **À retenir :** l'entropie décroît dès que la distribution devient inégale, car il est plus facile de deviner l'issue. L'entropie atteint 0 bit dans les cas extrêmes où l'issue est certaine d'avance.

Dans la suite du notebook, nous prolongerons ces notions avec l'entropie conditionnelle, l'information mutuelle et leur utilisation pratique pour analyser des données tabulaires.

### Entropie maximale et minimale

Pour $n$ issues possibles, l'entropie maximale est $\log_2(n)$ bits, atteinte lorsque les $n$ issues sont équiprobables. L'entropie minimale est 0 bit, lorsque l'une des issues a probabilité 1 (distribution dégénérée).

## 2. Code de Huffman : entropie et longueur moyenne d'un code optimal

L'entropie $H(X)$ représente une **limite théorique** (souvent fractionnaire) du nombre de bits nécessaires en moyenne pour coder les valeurs de $X$. Shannon a montré que l'on peut compresser sans perte l'information émise par une source $X$ jusqu'à environ $H(X)$ bits par symbole en moyenne (théorème du codage de source). Cependant, on ne peut pas descendre en dessous de $H(X)$ en moyenne.

Le **code de Huffman** est un algorithme de compression sans perte qui produit un code binaire à longueur variable optimal pour une distribution donnée. « Optimal » signifie que la longueur moyenne du code obtenu est la plus petite possible parmi tous les codes préfixes (décodables sans ambiguïté) pour cette distribution. Ce code sert d'illustration concrète : il atteint une longueur moyenne très proche de l'entropie de la source (toujours entre $H(X)$ et $H(X)+1$ en moyenne).

**Principe du codage de Huffman.** On attribue des codes binaires plus courts aux symboles les plus probables, et des codes plus longs aux symboles rares. L'algorithme fusionne itérativement les symboles les moins probables pour construire un arbre binaire optimal. Chaque feuille de l'arbre correspond à un symbole, et le chemin depuis la racine (0/1 à chaque branche) forme le code binaire de ce symbole.

Illustrons le codage de Huffman sur un petit exemple. Supposons une variable $X$ prenant 4 symboles $\{A, B, C, D\}$ de probabilités respectives $p(A)=0.4$, $p(B)=0.3$, $p(C)=0.2$, $p(D)=0.1$.


```python
probs = [0.4, 0.3, 0.2, 0.1]
H_X = entropy(probs)
print("Entropie H(X) =", H_X, "bits")
```

    Entropie H(X) = 1.8464393446710154 bits



```python
import heapq
from typing import Dict, Iterable, List, Optional

def huffman_codes(probabilities: Iterable[float], symbols: Optional[List[str]] = None) -> Dict[str, str]:
    # Construit un code de Huffman pour la liste de probabilités donnée.
    probabilities = list(probabilities)
    n = len(probabilities)
    if symbols is None:
        symbols = [chr(65 + i) for i in range(n)]
    heap = [(p, [s]) for p, s in zip(probabilities, symbols)]
    heapq.heapify(heap)
    codes = {s: '' for s in symbols}
    while len(heap) > 1:
        p1, sym1 = heapq.heappop(heap)
        p2, sym2 = heapq.heappop(heap)
        for s in sym1:
            codes[s] = '0' + codes[s]
        for s in sym2:
            codes[s] = '1' + codes[s]
        heapq.heappush(heap, (p1 + p2, sym1 + sym2))
    return codes

symbols = ['A', 'B', 'C', 'D']
codes = huffman_codes(probs, symbols)

print('Codes de Huffman :')
L_moy = 0
for s, p in zip(symbols, probs):
    code = codes[s]
    L_moy += len(code) * p
    print(f" Symbole {s}: code = {code} (longueur {len(code)})")

print(f"Longueur moyenne L = {L_moy:.3f} bits, Entropie H(X) = {H_X:.3f} bits")
```

    Codes de Huffman :
     Symbole A: code = 0 (longueur 1)
     Symbole B: code = 10 (longueur 2)
     Symbole C: code = 111 (longueur 3)
     Symbole D: code = 110 (longueur 3)
    Longueur moyenne L = 1.900 bits, Entropie H(X) = 1.846 bits


On constate que l'entropie $H(X) \approx 1.846$ bits est un plancher théorique : la longueur moyenne du code optimal est légèrement supérieure ($L \approx 1.90$ bits). Plus la distribution est déséquilibrée, plus $H(X)$ est faible et plus on gagne à utiliser un code variable. À l'extrême, si $p(A)=1$, on aurait $H(X)=0$ bit et Huffman attribuerait le code vide à $A$ (aucune incertitude).

En résumé, l'entropie quantifie l'information moyenne d'une variable, et les codes de longueur minimale (comme Huffman) réalisent en pratique cette compression de l'information. Un arbre de décision binaire optimal (posant des questions oui/non) aura une profondeur moyenne égale à l'entropie de la variable à deviner, dans le cas idéal.

---

## 3. Entropie conditionnelle et information mutuelle

Jusqu'ici, nous avons traité une seule variable. En data science, on étudie souvent la relation entre une **variable cible** $Y$ et des **variables explicatives** $X$. La théorie de l'information fournit des mesures pour quantifier ces relations.

**Entropie conjointe et conditionnelle.** Si l'on considère deux variables $X$ et $Y$, on peut définir l'entropie conjointe $H(X,Y)$ sur leur distribution commune, et l'entropie conditionnelle de $Y$ sachant $X$. L'entropie conditionnelle $H(Y \mid X)$ représente l'incertitude qui reste sur $Y$ quand on connaît la valeur de $X$. Formulée en probabilités :

$$
H(Y \mid X) = \sum_x P(X = x) \, H(Y \mid X = x).
$$

C'est la moyenne (pondérée par $P(X=x)$) des entropies de $Y$ dans chaque sous-population où $X=x$. On a toujours $H(Y \mid X) \le H(Y)$ : connaître $X$ ne peut pas augmenter l'incertitude sur $Y$ (au pire, $X$ est indépendant de $Y$ et on n'enlève aucune incertitude, $H(Y \mid X)=H(Y)$).

**Information mutuelle.** L'information mutuelle $I(X;Y)$ mesure la réduction d'incertitude sur $Y$ apportée par la connaissance de $X$ (et réciproquement, c'est symétrique). C'est une mesure de la **dépendance statistique** entre $X$ et $Y$. On peut la définir de plusieurs façons équivalentes, notamment :

$$
I(X;Y) = H(Y) - H(Y \mid X) = H(X) - H(X \mid Y).
$$

Autrement dit, $I(X;Y)$ est l'entropie de $Y$ dont on a été débarrassé en connaissant $X$. Si $X$ et $Y$ sont indépendants, connaître $X$ n'apprend rien sur $Y$ et l'information mutuelle est nulle. À l'inverse, si $X$ détermine parfaitement $Y$, l'incertitude sur $Y$ tombe à 0 avec $X$ et l'information mutuelle égale l'entropie $H(Y)$ entière.

Ces notions seront utiles pour analyser des jeux de données tabulaires : elles généralisent l'idée de « gain d'information » utilisé dans les arbres de décision et permettent de détecter des dépendances non linéaires entre variables. Dans la section suivante, nous les mettrons en pratique sur un jeu de données synthétique.
