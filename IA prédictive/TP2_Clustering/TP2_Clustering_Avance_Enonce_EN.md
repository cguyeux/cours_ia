# TP2 Advanced - Clustering: Modern Methods and Evaluation (approx. 3h)

[Back to contents](../../LISEZMOI.md)

## Objectives

- Master several clustering algorithms beyond K-Means
- Understand the strengths and weaknesses of each method
- Evaluate cluster quality with internal and external metrics
- Apply dimensionality reduction techniques for visualization
- Handle difficult cases: arbitrary shapes, varying densities

## Prerequisites

- Have completed the basic TP2 on K-Means
- Python 3.x, scikit-learn, matplotlib, numpy, pandas

```bash
pip install scikit-learn matplotlib numpy pandas umap-learn hdbscan
```

---

## Part 1 - Review and limits of K-Means (20 min)

1. Load the `Mall_Customers.csv` dataset
2. Apply K-Means with k=5 (review from the basic TP2)
3. **Identify the limitations**:
   - Sensitivity to initialization
   - Assumption of spherical clusters
   - Need to specify k in advance
   - Sensitivity to outliers

4. Generate a synthetic dataset with non-spherical clusters:

   ```python
   from sklearn.datasets import make_moons, make_blobs
   X_moons, y_moons = make_moons(n_samples=500, noise=0.1, random_state=42)
   ```

5. Test K-Means on this dataset and observe the failure

---

## Part 2 - Hierarchical Clustering (25 min)

### 2.1 Agglomerative (bottom-up)

1. Import `AgglomerativeClustering` from scikit-learn
2. Test different linkage metrics: `ward`, `complete`, `average`, `single`
3. Compare results on the Mall Customers dataset

### 2.2 Dendrogram

1. Use `scipy.cluster.hierarchy` to plot the dendrogram
2. Visually identify the optimal number of clusters
3. Cut the dendrogram at different heights

### 2.3 Pros/Cons

- No need to specify k in advance
- Interpretable visualization
- Complexity O(n^2) or O(n^3) depending on implementation

---

## Part 3 - DBSCAN: Density-Based Clustering (30 min)

### 3.1 Principle

- Clusters = dense regions separated by low-density regions
- Parameters: `eps` (neighborhood radius), `min_samples` (minimum density)

### 3.2 Application

1. Import `DBSCAN` from scikit-learn
2. Apply it to the `make_moons` dataset where K-Means failed
3. Compare the results

### 3.3 Hyperparameter choice

1. Plot the k-distance graph to estimate `eps`:

   ```python
   from sklearn.neighbors import NearestNeighbors
   neighbors = NearestNeighbors(n_neighbors=5)
   neighbors.fit(X)
   distances, _ = neighbors.kneighbors(X)
   distances = np.sort(distances[:, -1])
   plt.plot(distances)
   ```

2. Identify the "elbow" to choose `eps`

### 3.4 Outlier handling

- DBSCAN automatically labels outliers (label = -1)
- Analyze and interpret these outliers

---

## Part 4 - HDBSCAN: Hierarchical + Density (25 min)

### 4.1 Advantages over DBSCAN

- No need to choose `eps`
- Handles varying densities
- More robust to parameters

### 4.2 Application

```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
labels = clusterer.fit_predict(X)
```

### 4.3 Cluster probability analysis

- `clusterer.probabilities_`: membership confidence
- `clusterer.outlier_scores_`: anomaly score

---

## Part 5 - Gaussian Mixture Models (GMM) (25 min)

### 5.1 Principle

- Probabilistic modeling: each cluster is a Gaussian
- Soft clustering: membership probability for each cluster

### 5.2 Application

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
probas = gmm.predict_proba(X)
```

### 5.3 Selecting the number of components

- BIC (Bayesian Information Criterion)
- AIC (Akaike Information Criterion)

```python
bics = [GaussianMixture(n_components=k).fit(X).bic(X) for k in range(1, 11)]
```

### 5.4 Different covariance matrices

- `spherical`, `diag`, `tied`, `full`
- Impact on cluster shape

---

## Part 6 - Evaluation Metrics (30 min)

### 6.1 Internal metrics (no labels)

| Metric | Interpretation | scikit-learn function |
|--------|----------------|----------------------|
| **Silhouette Score** | -1 to 1, higher is better | `silhouette_score` |
| **Davies-Bouldin Index** | Lower is better | `davies_bouldin_score` |
| **Calinski-Harabasz Index** | Higher is better | `calinski_harabasz_score` |

1. Compute these 3 metrics for K-Means with k from 2 to 10
2. Plot the curves and identify the optimal k according to each metric
3. Compare with the elbow method (inertia)

### 6.2 External metrics (with true labels)

| Metric | Interpretation | Function |
|--------|----------------|----------|
| **Adjusted Rand Index (ARI)** | -1 to 1, 1 = perfect | `adjusted_rand_score` |
| **Normalized Mutual Information (NMI)** | 0 to 1, 1 = perfect | `normalized_mutual_info_score` |
| **Homogeneity / Completeness** | 0 to 1 | `homogeneity_score`, `completeness_score` |
| **V-measure** | Harmonic mean of H and C | `v_measure_score` |

1. Use the Iris dataset (with known labels)
2. Apply K-Means, DBSCAN, GMM
3. Compare the external metrics

### 6.3 Sample-level silhouette analysis

```python
from sklearn.metrics import silhouette_samples
sample_silhouette = silhouette_samples(X, labels)
```

- Identify poorly assigned points (negative silhouette)

---

## Part 7 - Dimensionality Reduction for Visualization (25 min)

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

- Note: t-SNE does not preserve global distances

### 7.3 UMAP (Uniform Manifold Approximation)

```python
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)
```

- Faster than t-SNE, better preservation of global structure

### 7.4 Visual comparison

- Create a 1x3 figure comparing PCA, t-SNE, UMAP colored by cluster

---

## Part 8 - Full case study (30 min)

### Dataset: Wholesale Customers

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)
```

### Full pipeline

1. **Exploration**: statistics, distributions, correlations
2. **Preprocessing**: normalization with `StandardScaler`
3. **Algorithm comparison**:
   - K-Means
   - Hierarchical clustering
   - DBSCAN
   - GMM
4. **Evaluation**: silhouette, Davies-Bouldin for each method
5. **Visualization**: UMAP with labels from the best method
6. **Interpretation**: average profile of each cluster

---

## Part 9 - Bonus exercises (optional)

1. **Spectral Clustering**: implement and compare on make_moons
2. **Mini-Batch K-Means**: compare speed vs quality on a large dataset
3. **Time series clustering**: use `tslearn` for clustering with DTW
