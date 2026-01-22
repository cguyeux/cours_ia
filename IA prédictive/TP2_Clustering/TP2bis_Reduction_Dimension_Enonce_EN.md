# TP2bis - Dimensionality Reduction: Theory and Practice (approx. 3h)

[Back to contents](../../LISEZMOI.md)

## Objectives

- Understand the motivations for dimensionality reduction
- Master Principal Component Analysis (PCA) and its variants
- Apply non-linear techniques: t-SNE, UMAP, Isomap
- Choose the right method for the context
- Interpret and visualize high-dimensional data

## Prerequisites

- Python 3.x, scikit-learn, matplotlib, numpy, pandas
- Basics of linear algebra (eigenvectors, eigenvalues)

```bash
pip install scikit-learn matplotlib numpy pandas umap-learn
```

---

## Part 1 - Motivation and the curse of dimensionality (20 min)

### 1.1 The curse of dimensionality

1. **Data sparsity**: generate n=100 points uniformly in:
   - A square [0,1]^2 (2D)
   - A cube [0,1]^3 (3D)
   - A hypercube [0,1]^10 (10D)

2. Compute the average distance to the nearest neighbor in each case

3. **Observe**: in high dimension, points are all "far" from each other

### 1.2 Why reduce dimensionality?

| Goal | Explanation |
|------|-------------|
| **Visualization** | Project to 2D/3D to explore data |
| **Denoising** | Remove noise in non-informative dimensions |
| **Compression** | Reduce memory footprint |
| **ML preprocessing** | Improve model performance/speed |
| **Avoid overfitting** | Fewer features = more generalizable models |

### 1.3 Method taxonomy

```
Dimensionality Reduction
|-- Linear
|   |-- PCA (unsupervised)
|   |-- LDA (supervised)
|   `-- Factor Analysis
`-- Non-linear (manifold learning)
    |-- t-SNE
    |-- UMAP
    |-- Isomap
    |-- LLE (Locally Linear Embedding)
    `-- MDS (Multidimensional Scaling)
```

---

## Part 2 - Principal Component Analysis (PCA) (40 min)

### 2.1 Mathematical principle

1. **Center the data**: subtract the mean of each feature
2. **Compute the covariance matrix**: Sigma = (1/n) X^T X
3. **Eigen decomposition**: Sigma v = lambda v
4. **Project**: eigenvectors with the largest eigenvalues = directions of maximum variance

### 2.2 PCA step by step (without scikit-learn)

```python
import numpy as np

# 1. Center the data
X_centered = X - X.mean(axis=0)

# 2. Covariance matrix
cov_matrix = np.cov(X_centered.T)

# 3. Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 4. Sort by decreasing eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. Project onto k components
k = 2
X_pca = X_centered @ eigenvectors[:, :k]
```

### 2.3 PCA with scikit-learn

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

### 2.4 Explained variance

1. Plot the **scree plot**: explained variance per component
2. Plot the **cumulative variance**: how many components for 95% variance?
3. Use `pca.explained_variance_ratio_`

### 2.5 Component interpretation

1. Analyze `pca.components_`: weight of each original feature
2. Create a **biplot**: projection + feature vectors
3. Identify the most contributive features

### 2.6 Application: Iris dataset

1. Apply PCA to the 4 Iris features
2. Visualize in 2D, colored by species
3. Interpret the first two components

---

## Part 3 - PCA variants (25 min)

### 3.1 Kernel PCA (non-linear PCA)

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)
```

1. Apply to the `make_circles` dataset
2. Compare linear PCA vs Kernel PCA (RBF)
3. Test different kernels: `poly`, `sigmoid`, `cosine`

### 3.2 Incremental PCA (for large datasets)

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=50, batch_size=200)
for batch in data_batches:
    ipca.partial_fit(batch)
```

- Useful when data does not fit in memory

### 3.3 Sparse PCA

```python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=5, alpha=1)
X_spca = spca.fit_transform(X)
```

- Imposes L1 regularization to obtain components with few non-zero features
- More interpretable

### 3.4 Randomized PCA

```python
pca = PCA(n_components=50, svd_solver='randomized')
```

- Fast approximation for large matrices
- Used automatically by scikit-learn if n_components << n_features

---

## Part 4 - t-SNE: Non-linear visualization (30 min)

### 4.1 Principle

1. Compute **similarities** between pairs of points in high dimension (Gaussians)
2. Compute similarities in low dimension (Student-t, hence the "t")
3. **Minimize KL divergence** between the two distributions
4. Result: points close in HD remain close in LD

### 4.2 Key hyperparameters

| Parameter | Effect |
|-----------|--------|
| `perplexity` | ~number of neighbors considered (typically 5-50) |
| `learning_rate` | Convergence speed (50-1000) |
| `n_iter` | Number of iterations (1000 recommended) |
| `early_exaggeration` | Amplifies clusters at the beginning |

### 4.3 Application

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)
```

### 4.4 Experiments

1. **MNIST/Digits**: visualize the 10 digits
2. **Vary perplexity**: compare 5, 30, 50, 100
3. **Observe the effect**: low perplexity = tight clusters, high perplexity = global structure

### 4.5 Limitations and precautions

Note: avoid over-interpreting t-SNE plots:

- Distances between clusters are NOT meaningful
- Cluster size does not reflect true density
- Results vary with `random_state`
- Do not use projected points for distance computations

---

## Part 5 - UMAP: Modern alternative to t-SNE (30 min)

### 5.1 Advantages over t-SNE

| Criterion | t-SNE | UMAP |
|----------|-------|------|
| Speed | Slow O(n^2) | Fast |
| Global structure | Poorly preserved | Better preserved |
| New points | Not possible | Possible (`transform`) |
| Reproducibility | Variable | Better |

### 5.2 Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `n_neighbors` | Local neighborhood size (5-50) |
| `min_dist` | Cluster compactness (0.0-1.0) |
| `metric` | Distance used (`euclidean`, `cosine`, etc.) |

### 5.3 Application

```python
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)
```

### 5.4 Experiments

1. Compare UMAP vs t-SNE on MNIST/Digits
2. Vary `n_neighbors`: 5, 15, 50, 200
3. Vary `min_dist`: 0.0, 0.1, 0.5, 0.99
4. **Project new points**:

   ```python
   reducer.fit(X_train)
   X_test_projected = reducer.transform(X_test)
   ```

### 5.5 Supervised UMAP

```python
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
X_umap_supervised = reducer.fit_transform(X, y=labels)
```

- Uses labels to guide the projection
- Clusters are more separated

---

## Part 6 - Other manifold learning methods (25 min)

### 6.1 Isomap (Isometric Mapping)

- Preserves **geodesic distances** (along the manifold)
- Builds a k-nearest-neighbors graph

```python
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2, n_neighbors=10)
X_isomap = isomap.fit_transform(X)
```

**Application**: Swiss Roll dataset

### 6.2 LLE (Locally Linear Embedding)

- Preserves **local linear relationships**
- Each point = linear combination of its neighbors

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_lle = lle.fit_transform(X)
```

### 6.3 MDS (Multidimensional Scaling)

- Preserves **distances between all points**
- Classical (Euclidean distances) or non-metric

```python
from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)
```

### 6.4 Comparison on Swiss Roll

1. Generate the Swiss Roll:

   ```python
   from sklearn.datasets import make_swiss_roll
   X, color = make_swiss_roll(n_samples=1500, noise=0.1)
   ```

2. Apply PCA, Isomap, LLE, t-SNE, UMAP
3. Compare results: which methods "unroll" the manifold correctly?

---

## Part 7 - Linear Discriminant Analysis (LDA) (20 min)

### 7.1 Difference from PCA

| PCA | LDA |
|-----|-----|
| Unsupervised | Supervised |
| Maximizes total variance | Maximizes class separability |
| n_components <= n_features | n_components <= n_classes - 1 |

### 7.2 Application

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

### 7.3 PCA vs LDA comparison

1. Apply to Iris (3 classes -> max 2 LDA components)
2. Visualize side by side
3. Observe that LDA separates classes better

### 7.4 LDA as a classifier

```python
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predictions = lda.predict(X_test)
```

---

## Part 8 - Method selection guide (15 min)

### 8.1 Decision tree

```
Goal?
|-- Exploratory visualization
|   |-- Small datasets (< 10k) -> t-SNE
|   `-- Large datasets -> UMAP
|-- ML preprocessing
|   |-- Linear reduction -> PCA
|   |-- Classification -> LDA
|   `-- Non-linear -> Kernel PCA or UMAP
|-- Feature interpretation -> PCA or Sparse PCA
`-- Compression/denoising -> PCA with n_components chosen by explained variance
```

### 8.2 Summary table

| Method | Type | Preserves | Speed | New points | Interpretable |
|--------|------|-----------|-------|------------|---------------|
| PCA | Linear | Global variance | Fast | Yes | Yes |
| Kernel PCA | Non-linear | Variance (kernel) | Medium | Yes | No |
| LDA | Supervised linear | Class separability | Fast | Yes | Yes |
| t-SNE | Non-linear | Local neighborhoods | Slow | No | No |
| UMAP | Non-linear | Local+global structure | Medium | Yes | No |
| Isomap | Non-linear | Geodesic distances | Medium | Yes | No |

---

## Part 9 - Full case study (25 min)

### Dataset: Fashion MNIST (simplified)

```python
from sklearn.datasets import fetch_openml
fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X, y = fashion.data[:5000], fashion.target[:5000].astype(int)
```

### Pipeline

1. **Exploration**: visualize a few images (28x28 pixels = 784 features)
2. **PCA**: reduce to 50 components, analyze explained variance
3. **t-SNE on PCA**: project the 50 PCA components to 2D
4. **Direct UMAP**: compare with t-SNE
5. **Post-reduction clustering**: K-Means on the UMAP projection
6. **Evaluation**: ARI between clusters and true labels

---

## Part 10 - Bonus exercises (optional)

1. **Autoencoder**: compare PCA with a simple autoencoder (via PyTorch/Keras)
2. **Trimap**: test this recent alternative to t-SNE/UMAP
3. **Dimensionality reduction for time series**: PCA on time-series data
4. **Feature extraction vs feature selection**: compare PCA with SelectKBest
5. **3D visualization**: project to 3D and plot with plotly

---

## Tips

- Always **standardize** before PCA (otherwise high-variance features dominate)
- For t-SNE/UMAP, apply PCA first if n_features >> 50
- t-SNE/UMAP visualizations are **qualitative**, not quantitative
- Keep `random_state` fixed for reproducibility

## Resources

- [Distill.pub - How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Scikit-learn Decomposition](https://scikit-learn.org/stable/modules/decomposition.html)

## Estimated duration

- 3 h (parts 1-9)
- +1 h for the bonus

---

[Access the solution](TP2bis_Reduction_Dimension_Corrige.ipynb)
