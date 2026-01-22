# Pandas Lab (Extended) (approx. 2h) - Iris Dataset

[Back to contents](../../LISEZMOI.md)

## Objectives

- Master the fundamental pandas operations (loading, exploration, manipulation)
- Know how to read/write CSV files
- Perform aggregations and group statistics
- Handle missing values
- Produce varied visualizations with matplotlib

## Prerequisites

- Python 3.x, pandas, scikit-learn, matplotlib, numpy

---

## Part 1 - Loading and exploration (20 min)

1. Import the Iris dataset from scikit-learn (`from sklearn.datasets import load_iris`)
2. Build a DataFrame with the columns: `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species`
3. Display the first 5 and last 5 rows (`head()`, `tail()`)
4. Display the DataFrame shape (`shape`)
5. Display `df.info()` and `df.describe()`
6. Display the data types (`dtypes`)

---

## Part 2 - Selection and indexing (20 min)

1. Select only the columns `SepalLengthCm` and `PetalLengthCm`
2. Use `loc[]` to select rows 10 to 20 (inclusive), columns `Species` and `PetalWidthCm`
3. Use `iloc[]` to select the first 5 rows, first 2 columns
4. Filter flowers where `SepalLengthCm > 6.0`
5. Filter with multiple conditions: `SepalLengthCm > 5.5` AND `PetalLengthCm < 4.0`
6. Use the `query()` method to filter the *versicolor* species

---

## Part 3 - Column and row manipulation (20 min)

1. Add a column `PetalRatio = PetalLengthCm / PetalWidthCm`
2. Add a column `SepalArea = SepalLengthCm * SepalWidthCm`
3. Create a categorical column `Taille`:
   - "small" if `SepalLengthCm < 5.0`
   - "medium" if `5.0 <= SepalLengthCm < 6.5`
   - "large" otherwise
4. Drop the `SepalArea` column
5. Drop rows where `SepalLengthCm < 5.0`
6. Sort the DataFrame by `PetalLengthCm` descending
7. Reset the index after deletions

---

## Part 4 - Missing values handling (15 min)

1. Introduce `NaN` in 5 random cells of the `PetalWidthCm` column:

   ```python
   import numpy as np
   df.loc[df.sample(5).index, "PetalWidthCm"] = np.nan
   ```

2. Count the number of missing values per column (`isna().sum()`)
3. Display rows containing `NaN` (`df[df.isna().any(axis=1)]`)
4. Replace `NaN` with the column mean (`fillna()`)
5. Drop rows containing `NaN` (on a copy, with `dropna()`)

---

## Part 5 - Aggregations and groupby (20 min)

1. Count occurrences per species (`value_counts()`)
2. Compute the mean of each numeric variable per species (`groupby().mean()`)
3. Compute multiple statistics per species: mean, std, min, max

   ```python
   df.groupby("Species").agg(["mean", "std", "min", "max"])
   ```

4. Compute the 25%, 50%, 75% percentiles of `PetalLengthCm` per species
5. Create a crosstab between `Species` and `Taille` (`pd.crosstab()`)

---

## Part 6 - Reading/writing files (10 min)

1. Export the DataFrame to CSV without the index:

   ```python
   df.to_csv("iris_enrichi.csv", index=False)
   ```

2. Reload the CSV file into a new DataFrame
3. Verify that the data are identical
4. Export to CSV with separator `;` and reload

---

## Part 7 - Visualizations (25 min)

### 7.1 Histogram

- Plot the histogram of `SepalLengthCm` (15 bins)

### 7.2 Simple scatter plot

- Plot `SepalLengthCm` vs `PetalLengthCm`

### 7.3 Scatter plot colored by species

- Same plot but with a different color per species and a legend

### 7.4 Boxplots by species

- Plot the boxplot of `PetalLengthCm` for each species

### 7.5 Bar chart

- Display the number of flowers per species

### 7.6 Correlation matrix

- Compute `df.corr()` on numeric columns
- Display as a heatmap with `plt.imshow()` or `plt.matshow()`

### 7.7 (Bonus) Multiple subplots

- Create a figure with 4 subplots (2x2) showing the histograms of the 4 numeric variables

---

## Part 8 - Bonus exercises (optional)

1. **Outlier**: Identify the flower with the largest `PetalRatio` - is it consistent?
2. **Advanced statistics**: Compute the coefficient of variation (std/mean) per species
3. **Apply**: Use `apply()` to create a column indicating if the flower is "typical" (all measures close to the mean of its species plus/minus 1 standard deviation)

---

## Tips

- Use `load_iris(as_frame=True)` to get a DataFrame directly
- For a colored scatter plot: create a dictionary `colors = {"setosa": "red", ...}` then map
- For the boxplot: `plt.boxplot()` with a list of series
- Use `plt.figure()` before each plot
- `plt.tight_layout()` avoids overlaps

## Estimated duration

- 2 h (parts 1-7)
- +30 min for the bonus

---

[Access the solution](TP_Pandas_Enrichi_Corrige.ipynb)
