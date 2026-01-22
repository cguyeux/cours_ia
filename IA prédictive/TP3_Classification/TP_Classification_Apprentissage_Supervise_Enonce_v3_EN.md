# TP - Machine Learning Classification

[Back to contents](../../LISEZMOI.md)

## Classification algorithms used

Before starting the lab, we present the algorithms that will be used. Each method has strengths, weaknesses, and key parameters to optimize. Diagrams and images are provided (links or auto-generated) to help understanding.

### Decision Tree

Principle: a decision tree splits the feature space through a series of questions like "is feature X greater than a threshold?". Each node corresponds to a decision and leaves correspond to predicted classes.

- **Advantages**: interpretable, easy to visualize, little preprocessing needed.
- **Drawbacks**: prone to overfitting if the tree is too deep.

**Key hyperparameters:**

- `max_depth`: limits the depth of the tree.
- `min_samples_split`: minimum number of samples to split a node.
- `min_samples_leaf`: minimum number of samples in a leaf.
- `criterion`: impurity measure (`entropy`, `gini`).

<https://www.youtube.com/watch?v=ZVR2Way4nwQ>

### Random Forest

Principle: a random forest is composed of many decision trees. Each tree is built on a subsample of the data and a subset of features. The final prediction is obtained by majority vote.

- **Advantages**: robust, reduces overfitting compared to a single tree, strong performance without complex tuning.
- **Drawbacks**: less interpretable than a single tree, more computationally expensive.

**Key hyperparameters:**

- `n_estimators`: number of trees.
- `max_depth`: maximum depth of the trees.
- `max_features`: number of features considered at each split.
- `min_samples_leaf`: minimum number of samples in a leaf.

<https://www.youtube.com/watch?v=v6VJ2RO66Ag>

### XGBoost (eXtreme Gradient Boosting)

Principle: boosting builds trees sequentially. Each new tree corrects the errors of the previous ones. XGBoost is an optimized and widely used implementation of gradient boosting.

- **Advantages**: often the best-performing algorithm on tabular data.
- **Drawbacks**: more complex tuning, training can be time-consuming.

**Key hyperparameters:**

- `n_estimators`: number of trees.
- `learning_rate`: learning rate (impact of each tree).
- `max_depth`: maximum depth.
- `subsample`: fraction of samples used per tree.
- `colsample_bytree`: fraction of features used per tree.

## Principle of `train_test_split`

When training a machine learning model, it is essential to evaluate performance on unseen data. We therefore split the dataset into two parts:

- **train set**: used to learn the model.
- **test set**: used only for the final evaluation.

Typical split: 80% for training and 20% for testing, while preserving class proportions (stratification).

<https://www.youtube.com/watch?v=SjOfbbfI2qY>

## Classification evaluation metrics

To judge the quality of a classification model, several metrics are used:

- **Accuracy**: overall proportion of correct predictions. Useful when classes are balanced, but can hide large errors on a minority class.
- **Precision**: among positive predictions, what proportion is truly positive? A precision of 80% means 20% of positive predictions are actually false positives.
- **Recall (sensitivity)**: among truly positive observations, what proportion is detected as positive by the model? A recall of 70% means 30% of positives were missed (false negatives).
- **F1-score**: harmonic mean of precision and recall \(F1 = 2 \times \frac{precision \times recall}{precision + recall}\). The harmonic mean yields a high score only if both precision and recall are good, since it penalizes very low values.

### Example: "cat" vs "Siamese cat"

If you search for "Siamese cat" images on Google:

- A generic query like "cat" will return many cat images, but most will not be Siamese. You get high recall (almost all Siamese cats are retrieved) but low precision (relevant results are drowned among other cat breeds).
- A query for "Siamese cat" will mostly return Siamese cats. Precision is high (few false positives), but recall can be lower if Google discards some relevant images that are mislabeled.

In an image search engine:

- **Maximizing recall** is crucial if the user prefers too many results rather than missing any (for example, a veterinarian who cannot miss any relevant image to diagnose a rare Siamese cat disease).
- **Maximizing precision** is preferred if the user values the quality of the first results (for example, a buyer who wants accurate Siamese cat photos for a listing).
- The **F1-score** balances these two goals: if precision or recall drops, the F1-score drops sharply. It is useful to optimize a model that must avoid false positives while not missing too many true positives.

<https://www.youtube.com/watch?v=Kdsp6soqA7o>

## Learning objectives

- Understand and apply several classification algorithms: Decision Tree, Random Forest, XGBoost.
- Know how to prepare a dataset, train, evaluate, and compare models.
- Analyze and interpret results (metrics, confusion matrices, feature importances).

## Prerequisites

- Python
- `pandas`
- `numpy`
- `matplotlib`
- Basics of `scikit-learn`

## Dataset used

The lab uses the **Breast Cancer Wisconsin** dataset from scikit-learn. The goal is to predict whether a tumor is benign or malignant based on cell measurements.

## Lab workflow

To help you progress step by step, the lab is split into operational stages. Each stage is an opportunity to add a new cell to your notebook and comment on the results.

1. **Prepare the work environment**
   - Create a new notebook or duplicate the provided template, then import the required libraries (`pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `sklearn`).
   - Load the Breast Cancer dataset from `sklearn.datasets` and convert it to a `DataFrame` for easier exploration.
   - Display the data dimensions, column names, and first rows to validate loading.
2. **Explore the dataset**
   - Check for missing values, examine class distribution, and compute a few descriptive statistics.
   - Visualize at least one pair of variables (via `pairplot` or `scatterplot`) to spot potential natural separations.
   - Identify strong correlations and note variables that may be redundant.
3. **Set up validation**
   - Split the data into training and test sets with `train_test_split`, stratifying on the target variable.
   - Normalize features (StandardScaler or MinMaxScaler) if needed. Keep the raw version if you want to compare.
   - Record the size of each set and justify the train/test ratio.
4. **Establish a baseline model (Decision Tree)**
   - Train a simple decision tree to have a comparison point.
   - Compute the main metrics on train and test, then briefly discuss model quality.
   - Display the confusion matrix and, if possible, the tree or learned rules.
5. **Improve performance (Random Forest)**
   - Train a Random Forest starting with default hyperparameters.
   - Test at least two different configurations (for example number of trees or maximum depth) and record results in a comparison table.
   - Analyze feature importance from the model and comment on the top three most influential variables.
6. **Explore a more advanced model (XGBoost)**
   - Install/import `xgboost` if needed and train a first model with standard parameters.
   - Gradually adjust `learning_rate`, `max_depth`, or `n_estimators` and observe the impact on test metrics.
   - Document training times or difficulties encountered (parameter handling, convergence, etc.).
7. **Compare and interpret**
   - Gather the main metrics (accuracy, precision, recall, f1-score) for the three models in a summary table and discuss differences.
   - Plot the confusion matrices and highlight common or algorithm-specific error patterns.
   - Describe, using feature importances, which characteristics contribute the most to the decision.
8. **Conclude and open up**
   - Write a short summary explaining which model you would choose for production and why.
   - List possible improvement axes: additional tuning, cross-validation, class imbalance handling, advanced interpretability, etc.
   - Ensure the notebook is clear (titles, comments, conclusions) and ready for submission.

## Deliverables

- A completed Jupyter notebook (code + written answers to questions).
- A short report (5-10 lines) comparing the models and concluding which one seems most appropriate.
