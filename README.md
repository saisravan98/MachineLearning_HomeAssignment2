# C5710-Machine-Learning-Home-Assignment-2

NAME - SAI SRAVAN CHINTALA #700773836 

# Question 7. Decision Tree on Iris — Depth Sweep (scikit-learn)

This repo contains a minimal experiment that trains a `DecisionTreeClassifier` on the classic **Iris** dataset, compares three tree complexities (`max_depth = 1, 2, 3`), and reports **training** and **test** accuracy. It’s a compact demo of model capacity vs. generalization (underfitting ↔ overfitting).

---

## What’s inside

- Loads the Iris dataset from `sklearn.datasets`
- Stratified train/test split (80/20) with a fixed `random_state`
- Trains decision trees with `max_depth ∈ {1,2,3}`
- Prints per-depth **train** and **test** accuracy
- Notes on how to interpret the results (signs of under/overfitting)

---

## Requirements

- Python 3.8+
- scikit-learn ≥ 1.2
- numpy ≥ 1.22



# Questionn 8. kNN on Iris (2 features) — Decision Boundaries for k = 1, 3, 5, 10

This script trains **k-Nearest Neighbors (kNN)** classifiers on the **Iris** dataset using **only two features** — *sepal length* and *sepal width* — and plots the **decision boundaries** for `k ∈ {1, 3, 5, 10}`. It demonstrates how the choice of **k** controls model complexity (jagged vs. smooth boundaries) and generalization.  


---

## What the code does

- Loads **Iris** and keeps columns 0–1 (sepal length/width) to make a 2D plane for visualization.  
- Trains `KNeighborsClassifier` with `metric="euclidean"` for `k = 1, 3, 5, 10`. (The slides give the Euclidean distance formula and a worked example.)   
- Builds a dense meshgrid over the feature space and predicts the class on each grid point, then **contour-fills** the regions and overlays the training points.  
- Produces a 2×2 figure comparing the boundaries across k. (Iris kNN visuals are shown in the deck.) 

---

## Why this example

- **k = 1** tends to **overfit** with very **jagged** boundaries; as **k increases**, the boundary **smooths**, trading variance for bias. This “depends on k” behavior is highlighted in the slides.  
- kNN is **nonparametric**: there’s no training phase beyond storing the data; prediction uses neighbors + majority vote. 

---

# Question 9. kNN (k=5) on Iris — Confusion Matrix, Metrics, and ROC/AUC

This script trains a **k-Nearest Neighbors** classifier (`k=5`) on the **Iris** dataset, then evaluates performance by:
1) displaying the **confusion matrix**,  
2) printing **accuracy, precision, recall, F1** via `classification_report`, and  
3) plotting **one-vs-rest ROC curves** with **per-class AUC**, plus **micro** and **macro** AUC.

---

## What the code does

- Loads Iris and makes a **stratified** train/test split (`test_size=0.3`, `random_state=42`).
- Fits `KNeighborsClassifier(n_neighbors=5, metric="euclidean")`.
- **Confusion matrix:** uses `confusion_matrix` + `ConfusionMatrixDisplay`.  
- **Classification report:** prints per-class precision/recall/F1 and macro/weighted averages. 
- **ROC/AUC (multiclass):** one-vs-rest using `label_binarize` and `predict_proba`; plots per-class ROC and reports AUC, plus micro-average and macro-average.

---

## Requirements

- Python 3.8+
- `scikit-learn`, `numpy`, `matplotlib`

```bash
pip install scikit-learn numpy matplotlib


