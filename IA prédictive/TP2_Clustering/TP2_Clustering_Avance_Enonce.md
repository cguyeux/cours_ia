# TP2 Avancé — Clustering : Méthodes Modernes et Évaluation (≈ 3h)

[⬅️ Retour au sommaire](../../LISEZMOI.md)

## Objectifs

- Maîtriser plusieurs algorithmes de clustering au-delà de K-Means
- Comprendre les forces et faiblesses de chaque méthode
- Évaluer la qualité des clusters avec des métriques internes et externes
- Appliquer des techniques de réduction de dimension pour la visualisation
- Gérer les cas difficiles : clusters de formes arbitraires, densités variables

## Prérequis

- Avoir suivi le TP2 de base sur K-Means
- Python 3.x, scikit-learn, matplotlib, numpy, pandas

```bash
pip install scikit-learn matplotlib numpy pandas umap-learn hdbscan
```

---

## Partie 1 — Rappel et limites de K-Means (20 min)

1. Charger le dataset `Mall_Customers.csv`
2. Appliquer K-Means avec k=5 (rappel du TP2 de base)
3. **Identifier les limites** :
   - Sensibilité à l'initialisation
   - Hypothèse de clusters sphériques
   - Nécessité de spécifier k à l'avance
   - Sensibilité aux outliers

4. Générer un dataset synthétique avec des clusters non sphériques :

   ```python
   from sklearn.datasets import make_moons, make_blobs
   X_moons, y_moons = make_moons(n_samples=500, noise=0.1, random_state=42)
   ```

5. Tester K-Means sur ce dataset et observer l'échec

---

## Partie 2 — Clustering Hiérarchique (25 min)

### 2.1 Agglomératif (bottom-up)

1. Importer `AgglomerativeClustering` de scikit-learn
2. Tester différentes métriques de liaison : `ward`, `complete`, `average`, `single`
3. Comparer les résultats sur le dataset Mall Customers

### 2.2 Dendrogramme

1. Utiliser `scipy.cluster.hierarchy` pour tracer le dendrogramme
2. Identifier visuellement le nombre optimal de clusters
3. Couper le dendrogramme à différentes hauteurs

### 2.3 Avantages/Inconvénients

- Pas besoin de spécifier k à l'avance
- Visualisation interprétable
- Complexité O(n²) ou O(n³) selon l'implémentation

---

## Partie 3 — DBSCAN : Clustering par Densité (30 min)

### 3.1 Principe

- Clusters = régions denses séparées par des régions de faible densité
- Paramètres : `eps` (rayon de voisinage), `min_samples` (densité minimale)

### 3.2 Application

1. Importer `DBSCAN` de scikit-learn
2. Appliquer sur le dataset `make_moons` où K-Means échouait
3. Comparer les résultats

### 3.3 Choix des hyperparamètres

1. Tracer le graphe des k-distances pour estimer `eps` :

   ```python
   from sklearn.neighbors import NearestNeighbors
   neighbors = NearestNeighbors(n_neighbors=5)
   neighbors.fit(X)
   distances, _ = neighbors.kneighbors(X)
   distances = np.sort(distances[:, -1])
   plt.plot(distances)
   ```

2. Identifier le "coude" pour choisir `eps`

### 3.4 Gestion des outliers

- DBSCAN identifie automatiquement les points aberrants (label = -1)
- Analyser et interpréter ces outliers

---

## Partie 4 — HDBSCAN : Hiérarchique + Densité (25 min)

### 4.1 Avantages sur DBSCAN

- Pas besoin de choisir `eps`
- Gère les densités variables
- Plus robuste aux paramètres

### 4.2 Application

```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
labels = clusterer.fit_predict(X)
```

### 4.3 Analyse des probabilités de cluster

- `clusterer.probabilities_` : confiance d'appartenance
- `clusterer.outlier_scores_` : score d'anomalie

---

## Partie 5 — Gaussian Mixture Models (GMM) (25 min)

### 5.1 Principe

- Modélisation probabiliste : chaque cluster est une gaussienne
- Clustering soft : probabilité d'appartenance à chaque cluster

### 5.2 Application

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
probas = gmm.predict_proba(X)
```

### 5.3 Sélection du nombre de composantes

- Critère BIC (Bayesian Information Criterion)
- Critère AIC (Akaike Information Criterion)

```python
bics = [GaussianMixture(n_components=k).fit(X).bic(X) for k in range(1, 11)]
```

### 5.4 Différentes matrices de covariance

- `spherical`, `diag`, `tied`, `full`
- Impact sur la forme des clusters

---

## Partie 6 — Métriques d'Évaluation (30 min)

### 6.1 Métriques internes (sans labels)

| Métrique | Interprétation | Fonction scikit-learn |
|----------|----------------|----------------------|
| **Silhouette Score** | -1 à 1, plus c'est haut mieux c'est | `silhouette_score` |
| **Davies-Bouldin Index** | Plus c'est bas mieux c'est | `davies_bouldin_score` |
| **Calinski-Harabasz Index** | Plus c'est haut mieux c'est | `calinski_harabasz_score` |

1. Calculer ces 3 métriques pour K-Means avec k de 2 à 10
2. Tracer les courbes et identifier le k optimal selon chaque métrique
3. Comparer avec la méthode du coude (inertie)

### 6.2 Métriques externes (avec labels réels)

| Métrique | Interprétation | Fonction |
|----------|----------------|----------|
| **Adjusted Rand Index (ARI)** | -1 à 1, 1 = parfait | `adjusted_rand_score` |
| **Normalized Mutual Information (NMI)** | 0 à 1, 1 = parfait | `normalized_mutual_info_score` |
| **Homogeneity / Completeness** | 0 à 1 | `homogeneity_score`, `completeness_score` |
| **V-measure** | Moyenne harmonique H et C | `v_measure_score` |

1. Utiliser le dataset Iris (avec labels connus)
2. Appliquer K-Means, DBSCAN, GMM
3. Comparer les métriques externes

### 6.3 Analyse de la silhouette par échantillon

```python
from sklearn.metrics import silhouette_samples
sample_silhouette = silhouette_samples(X, labels)
```

- Identifier les points mal classés (silhouette négative)

---

## Partie 7 — Réduction de Dimension pour Visualisation (25 min)

### 7.1 PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

### 7.2 t-SNE

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)
```

- Attention : t-SNE ne préserve pas les distances globales

### 7.3 UMAP (Uniform Manifold Approximation)

```python
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)
```

- Plus rapide que t-SNE, meilleure préservation de la structure globale

### 7.4 Comparaison visuelle

- Créer une figure 1×3 comparant PCA, t-SNE, UMAP colorés par cluster

---

## Partie 8 — Cas d'étude complet (30 min)

### Dataset : Wholesale Customers

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)
```

### Pipeline complet

1. **Exploration** : statistiques, distributions, corrélations
2. **Prétraitement** : normalisation avec `StandardScaler`
3. **Comparaison d'algorithmes** :
   - K-Means
   - Clustering hiérarchique
   - DBSCAN
   - GMM
4. **Évaluation** : silhouette, Davies-Bouldin pour chaque méthode
5. **Visualisation** : UMAP avec les labels de la meilleure méthode
6. **Interprétation** : profil moyen de chaque cluster

---

## Partie 9 — Exercices bonus (optionnel)

1. **Spectral Clustering** : implémenter et comparer sur make_moons
2. **Mini-Batch K-Means** : comparer vitesse vs qualité sur un grand dataset
3. **Clustering de séries temporelles** : utiliser `tslearn` pour clustering avec DTW
4. **Clustering de texte** : TF-IDF + K-Means sur un corpus de documents
5. **Consensus clustering** : combiner plusieurs méthodes de clustering

---

## Récapitulatif des algorithmes

| Algorithme | Forme des clusters | Besoin de k | Gère outliers | Complexité |
|------------|-------------------|-------------|---------------|------------|
| K-Means | Sphérique | Oui | Non | O(n·k·i) |
| Hiérarchique | Quelconque | Non (dendro) | Non | O(n²) |
| DBSCAN | Arbitraire | Non | Oui | O(n log n) |
| HDBSCAN | Arbitraire | Non | Oui | O(n log n) |
| GMM | Elliptique | Oui | Non | O(n·k) |

---

## Conseils

- Toujours normaliser les données avant clustering
- Tester plusieurs algorithmes et comparer les métriques
- La visualisation est essentielle pour l'interprétation
- Le "bon" nombre de clusters dépend du contexte métier

## Ressources

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)

## Durée estimée

- 3h (parties 1-8)
- +1h pour les bonus

---

📘 **[Accéder au corrigé](TP2_Clustering_Avance_Corrige.ipynb)**
