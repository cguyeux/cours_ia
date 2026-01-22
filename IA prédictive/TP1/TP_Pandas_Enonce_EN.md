# Pandas Lab (approx. 1h) - Iris Dataset

[Back to contents](../../LISEZMOI.md)

## Objectives

- Download an existing dataset (Iris via scikit-learn), load it into pandas, then perform manipulations and visualizations with matplotlib.

## Prerequisites

- Python 3.x, pandas, scikit-learn, matplotlib.

## Instructions

- Import the Iris dataset from scikit-learn (`from sklearn.datasets import load_iris`).
- Build a pandas DataFrame with the columns: sepal_length, sepal_width, petal_length, petal_width, species.
- Perform common manipulations (statistics, filtering, renaming, adding/removing columns/rows).
- Produce plots with matplotlib (no seaborn required).

## Required work

- Load Iris and display the first 5 rows plus the DataFrame shape.
- Display `df.info()` and `df.describe()`.
- Rename the columns to `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species`.
- Add the column `PetalRatio = PetalLengthCm / PetalWidthCm` and `SepalRatio = SepalLengthCm * SepalWidthCm`.
- Delete the `SepalRatio` column.
- Delete rows where `SepalLengthCm < 5.0`.
- Filter only rows from the *setosa* species.
- Count the number of occurrences by species (frequency table).
- Visualize:
  1. a histogram of a numeric variable;
  2. a scatter plot between two variables;
  3. a boxplot by species;
  4. a bar chart of the count per species.

## Tips

- Use `load_iris(as_frame=True)` to get a DataFrame easily.
- For the scatter: `plt.scatter(x, y)` then `plt.xlabel`, `plt.ylabel`, `plt.title`.
- For the boxplot: `plt.boxplot` with a list of series (one per species).
- Use `plt.figure()` before each plot to avoid overlaps.

## Estimated duration

- 1 h.
