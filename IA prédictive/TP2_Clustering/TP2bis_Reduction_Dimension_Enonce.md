# TP2bis — Réduction de Dimension : Théorie et Pratique (≈ 3h)

[⬅️ Retour au sommaire](../../LISEZMOI.md)

## Objectifs

- Comprendre les motivations de la réduction de dimension
- Maîtriser l'Analyse en Composantes Principales (PCA) et ses variantes
- Appliquer des techniques non-linéaires : t-SNE, UMAP, Isomap
- Savoir choisir la bonne méthode selon le contexte
- Interpréter et visualiser des données haute dimension

## Prérequis

- Python 3.x, scikit-learn, matplotlib, numpy, pandas
- Notions d'algèbre linéaire (vecteurs propres, valeurs propres)

```bash
pip install scikit-learn matplotlib numpy pandas umap-learn
```

---

## Partie 1 — Motivation et fléau de la dimension (20 min)

### 1.1 Le fléau de la dimensionnalité (Curse of Dimensionality)

1. **Sparsité des données** : générer n=100 points uniformément dans :
   - Un carré [0,1]² (2D)
   - Un cube [0,1]³ (3D)
   - Un hypercube [0,1]^10 (10D)

2. Calculer la distance moyenne au plus proche voisin dans chaque cas

3. **Observer** : en haute dimension, les points sont tous "loin" les uns des autres

### 1.2 Pourquoi réduire la dimension ?

| Objectif | Explication |
|----------|-------------|
| **Visualisation** | Projeter en 2D/3D pour explorer les données |
| **Débruitage** | Éliminer le bruit dans les dimensions non informatives |
| **Compression** | Réduire la taille mémoire |
| **Pré-traitement ML** | Améliorer la performance/vitesse des modèles |
| **Éviter le surapprentissage** | Moins de features = modèles plus généralisables |

### 1.3 Taxonomie des méthodes

```
Réduction de Dimension
├── Linéaires
│   ├── PCA (non supervisé)
│   ├── LDA (supervisé)
│   └── Analyse Factorielle
└── Non-linéaires (Manifold Learning)
    ├── t-SNE
    ├── UMAP
    ├── Isomap
    ├── LLE (Locally Linear Embedding)
    └── MDS (Multidimensional Scaling)
```

---

## Partie 2 — Analyse en Composantes Principales (PCA) (40 min)

### 2.1 Principe mathématique

1. **Centrer les données** : soustraire la moyenne de chaque feature
2. **Calculer la matrice de covariance** : Σ = (1/n) X^T X
3. **Décomposer en vecteurs/valeurs propres** : Σv = λv
4. **Projeter** : les vecteurs propres avec les plus grandes valeurs propres = directions de variance maximale

### 2.2 PCA pas à pas (sans scikit-learn)

```python
import numpy as np

# 1. Centrer les données
X_centered = X - X.mean(axis=0)

# 2. Matrice de covariance
cov_matrix = np.cov(X_centered.T)

# 3. Décomposition en valeurs propres
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 4. Trier par valeur propre décroissante
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. Projeter sur k composantes
k = 2
X_pca = X_centered @ eigenvectors[:, :k]
```

### 2.3 PCA avec scikit-learn

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

### 2.4 Variance expliquée

1. Tracer le **scree plot** : variance expliquée par composante
2. Tracer la **variance cumulée** : combien de composantes pour 95% de variance ?
3. Utiliser `pca.explained_variance_ratio_`

### 2.5 Interprétation des composantes

1. Analyser `pca.components_` : poids de chaque feature originale
2. Créer un **biplot** : projection + vecteurs des features
3. Identifier les features les plus contributives

### 2.6 Application : Dataset Iris

1. Appliquer PCA sur les 4 features d'Iris
2. Visualiser en 2D, coloré par espèce
3. Interpréter les 2 premières composantes

---

## Partie 3 — Variantes de PCA (25 min)

### 3.1 Kernel PCA (PCA non-linéaire)

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)
```

1. Appliquer sur le dataset `make_circles`
2. Comparer PCA linéaire vs Kernel PCA (RBF)
3. Tester différents noyaux : `poly`, `sigmoid`, `cosine`

### 3.2 Incremental PCA (pour grands datasets)

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=50, batch_size=200)
for batch in data_batches:
    ipca.partial_fit(batch)
```

- Utile quand les données ne tiennent pas en mémoire

### 3.3 Sparse PCA

```python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=5, alpha=1)
X_spca = spca.fit_transform(X)
```

- Impose une régularisation L1 pour avoir des composantes avec peu de features non nulles
- Plus interprétable

### 3.4 Randomized PCA

```python
pca = PCA(n_components=50, svd_solver='randomized')
```

- Approximation rapide pour les grandes matrices
- Utilisé automatiquement par scikit-learn si n_components << n_features

---

## Partie 4 — t-SNE : Visualisation non-linéaire (30 min)

### 4.1 Principe

1. Calculer des **similarités** entre paires de points en haute dimension (gaussiennes)
2. Calculer des similarités en basse dimension (Student-t, d'où le "t")
3. **Minimiser la divergence KL** entre les deux distributions
4. Résultat : les points proches en HD restent proches en BD

### 4.2 Hyperparamètres clés

| Paramètre | Effet |
|-----------|-------|
| `perplexity` | ~nombre de voisins considérés (5-50 typiquement) |
| `learning_rate` | Vitesse de convergence (50-1000) |
| `n_iter` | Nombre d'itérations (1000 min recommandé) |
| `early_exaggeration` | Amplifie les clusters au début |

### 4.3 Application

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)
```

### 4.4 Expérimentations

1. **Dataset MNIST/Digits** : visualiser les 10 chiffres
2. **Varier perplexity** : comparer 5, 30, 50, 100
3. **Observer l'effet** : perplexity bas = clusters serrés, perplexity haut = structure globale

### 4.5 Limites et précautions

⚠️ **Attention aux interprétations abusives** :

- Les distances entre clusters ne sont PAS significatives
- La taille des clusters ne reflète pas leur densité réelle
- Les résultats varient selon `random_state`
- Ne pas utiliser pour des calculs de distance après projection

---

## Partie 5 — UMAP : Alternative moderne à t-SNE (30 min)

### 5.1 Avantages sur t-SNE

| Critère | t-SNE | UMAP |
|---------|-------|------|
| Vitesse | Lent O(n²) | Rapide |
| Structure globale | Mal préservée | Mieux préservée |
| Nouveaux points | Impossible | Possible (`transform`) |
| Reproductibilité | Variable | Meilleure |

### 5.2 Hyperparamètres

| Paramètre | Effet |
|-----------|-------|
| `n_neighbors` | Taille du voisinage local (5-50) |
| `min_dist` | Compacité des clusters (0.0-1.0) |
| `metric` | Distance utilisée (`euclidean`, `cosine`, etc.) |

### 5.3 Application

```python
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)
```

### 5.4 Expérimentations

1. Comparer UMAP vs t-SNE sur MNIST/Digits
2. Varier `n_neighbors` : 5, 15, 50, 200
3. Varier `min_dist` : 0.0, 0.1, 0.5, 0.99
4. **Projection de nouveaux points** :

   ```python
   reducer.fit(X_train)
   X_test_projected = reducer.transform(X_test)
   ```

### 5.5 UMAP supervisé

```python
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
X_umap_supervised = reducer.fit_transform(X, y=labels)
```

- Utilise les labels pour guider la projection
- Clusters plus séparés

---

## Partie 6 — Autres méthodes de Manifold Learning (25 min)

### 6.1 Isomap (Isometric Mapping)

- Préserve les **distances géodésiques** (le long de la variété)
- Construit un graphe de k plus proches voisins

```python
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2, n_neighbors=10)
X_isomap = isomap.fit_transform(X)
```

**Application** : dataset Swiss Roll

### 6.2 LLE (Locally Linear Embedding)

- Préserve les **relations linéaires locales**
- Chaque point = combinaison linéaire de ses voisins

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_lle = lle.fit_transform(X)
```

### 6.3 MDS (Multidimensional Scaling)

- Préserve les **distances entre tous les points**
- Classique (distances euclidiennes) ou non-métrique

```python
from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)
```

### 6.4 Comparaison sur Swiss Roll

1. Générer le Swiss Roll :

   ```python
   from sklearn.datasets import make_swiss_roll
   X, color = make_swiss_roll(n_samples=1500, noise=0.1)
   ```

2. Appliquer PCA, Isomap, LLE, t-SNE, UMAP
3. Comparer les résultats : qui "déroule" correctement la variété ?

---

## Partie 7 — Analyse Discriminante Linéaire (LDA) (20 min)

### 7.1 Différence avec PCA

| PCA | LDA |
|-----|-----|
| Non supervisé | Supervisé |
| Maximise la variance totale | Maximise la séparabilité des classes |
| Peut avoir n_components ≤ n_features | n_components ≤ n_classes - 1 |

### 7.2 Application

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

### 7.3 Comparaison PCA vs LDA

1. Appliquer sur Iris (3 classes → max 2 composantes LDA)
2. Visualiser côte à côte
3. Observer que LDA sépare mieux les classes

### 7.4 LDA comme classifieur

```python
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predictions = lda.predict(X_test)
```

---

## Partie 8 — Guide de choix des méthodes (15 min)

### 8.1 Arbre de décision

```
But ?
├── Visualisation exploratoire
│   ├── Petits datasets (< 10k) → t-SNE
│   └── Grands datasets → UMAP
├── Pré-traitement ML
│   ├── Réduction linéaire → PCA
│   ├── Classification → LDA
│   └── Non-linéaire → Kernel PCA ou UMAP
├── Interprétation des features → PCA ou Sparse PCA
└── Compression/débruitage → PCA avec n_components choisi par variance expliquée
```

### 8.2 Tableau récapitulatif

| Méthode | Type | Préserve | Vitesse | Nouveaux pts | Interprétable |
|---------|------|----------|---------|--------------|---------------|
| PCA | Linéaire | Variance globale | ⚡⚡⚡ | ✅ | ✅ |
| Kernel PCA | Non-linéaire | Variance (kernel) | ⚡⚡ | ✅ | ❌ |
| LDA | Linéaire supervisé | Séparabilité classes | ⚡⚡⚡ | ✅ | ✅ |
| t-SNE | Non-linéaire | Voisinages locaux | ⚡ | ❌ | ❌ |
| UMAP | Non-linéaire | Structure locale+globale | ⚡⚡ | ✅ | ❌ |
| Isomap | Non-linéaire | Distances géodésiques | ⚡⚡ | ✅ | ❌ |

---

## Partie 9 — Cas d'étude complet (25 min)

### Dataset : Fashion MNIST (simplifié)

```python
from sklearn.datasets import fetch_openml
fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X, y = fashion.data[:5000], fashion.target[:5000].astype(int)
```

### Pipeline

1. **Exploration** : visualiser quelques images (28×28 pixels = 784 features)
2. **PCA** : réduire à 50 composantes, analyser la variance expliquée
3. **t-SNE sur PCA** : projeter les 50 composantes PCA en 2D
4. **UMAP direct** : comparer avec t-SNE
5. **Clustering post-réduction** : K-Means sur la projection UMAP
6. **Évaluation** : ARI entre clusters et labels réels

---

## Partie 10 — Exercices bonus (optionnel)

1. **Autoencoder** : comparer PCA avec un autoencoder simple (via PyTorch/Keras)
2. **Trimap** : tester cette alternative récente à t-SNE/UMAP
3. **Réduction de dimension pour séries temporelles** : PCA sur données temporelles
4. **Feature Extraction vs Feature Selection** : comparer PCA avec SelectKBest
5. **Visualisation 3D** : projeter en 3D et afficher avec plotly

---

## Conseils

- Toujours **standardiser** avant PCA (sinon les features à grande variance dominent)
- Pour t-SNE/UMAP, appliquer d'abord PCA si n_features >> 50
- Les visualisations t-SNE/UMAP sont **qualitatives**, pas quantitatives
- Garder le `random_state` fixé pour reproductibilité

## Ressources

- [Distill.pub - How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Scikit-learn Decomposition](https://scikit-learn.org/stable/modules/decomposition.html)

## Durée estimée

- 3h (parties 1-9)
- +1h pour les bonus

---

📘 **[Accéder au corrigé](TP2bis_Reduction_Dimension_Corrige.ipynb)**
