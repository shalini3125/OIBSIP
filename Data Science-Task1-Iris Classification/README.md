Task 1 · Iris Flower Classification

📌 Objective

Train a machine learning classification model to identify the species of an iris flower — Setosa, Versicolor, or Virginica — based on four physical measurements: sepal length, sepal width, petal length, and petal width.

🛠️ Tech Stack


Python
scikit-learn
pandas
matplotlib / seaborn
Jupyter Notebook


📂 Dataset

The Iris dataset is loaded directly from sklearn.datasets.load_iris() — no external download required. It contains 150 samples, 50 for each of the 3 species, with 4 numeric features per sample.

🔍 Project Workflow


Load the dataset using load_iris() and convert it into a pandas DataFrame
Exploratory Data Analysis (EDA) — shape, data types, null value check, descriptive statistics
Visualisations — pairplot/scatter matrix by species, box plots per feature, correlation heatmap
Feature selection discussion — identifying which features are most discriminative
Train/test split — 80/20 split using train_test_split, with feature scaling via StandardScaler
Model training — 4 classifiers trained and compared:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest



Model evaluation — accuracy score, confusion matrix, and classification report (precision, recall, F1-score) for each model
Best model declaration — models compared on test accuracy, with written justification for the final choice
Model deployment prep — best model, scaler, and target names saved using joblib (.pkl files) for future use outside the notebook


📊 Key Results


All 4 models achieve high accuracy (often 93–100%) on the test set, since the Iris dataset is small and the species are well separated, especially by petal measurements.
Petal length and petal width are the most discriminative features (little to no overlap between species); sepal width is the weakest predictor on its own.
The best-performing model is declared in the notebook based on test accuracy, with ties broken in favor of Random Forest for its robustness against overfitting.
