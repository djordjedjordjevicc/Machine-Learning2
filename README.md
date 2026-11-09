# Water Quality Classification – Machine Learning

A supervised learning project that predicts whether a water sample is **safe to drink** from the concentrations of 20 chemical and biological substances. It compares 7 classification algorithms, looks at how class imbalance affects them, and tests the effect of feature selection and hyperparameter tuning.

All work is in [`Masinsko1.ipynb`](Masinsko1.ipynb). Notebook comments are in Serbian.

## Dataset

[`waterQuality1.csv`](waterQuality1.csv) is a public water quality dataset (Kaggle):

- **7,999 samples, 20 numeric features:** aluminium, ammonia, arsenic, barium, cadmium, chloramine, chromium, copper, fluoride, bacteria, viruses, lead, nitrates, nitrites, mercury, perchlorate, radium, selenium, silver, uranium
- **Target:** `is_safe` (1 = safe, 0 = not safe)
- **Strongly imbalanced:** only **11.4%** of samples are safe (912 vs 7,084)

**Cleaning:** 3 rows had an invalid target value (`#NUM!`) and were removed. There are no missing values or duplicates otherwise.

## Approach

### 1. Exploratory data analysis
- Distribution of every feature (histograms, skewness, kurtosis) and boxplots, overall and per class
- Pearson and Spearman correlation matrices, plus each feature's correlation with the target
- PCA projection to 2D to see how well the classes separate
- Full descriptive statistics (mean, median, mode, geometric and harmonic mean, variance, IQR, and so on)

### 2. Feature selection
Two methods are combined:
- **Mutual Information** with the target
- **ExtraTreesClassifier** feature importance

Taking the union of the top features from both methods gives two reduced datasets: **6 features** (top 5 from each) and **10 features** (top 8 from each). Aluminium, cadmium, arsenic, chloramine and perchlorate rank highest under both methods.

### 3. Models
Random Forest · Decision Tree · Logistic Regression · KNN · Naive Bayes · SVM (RBF) · XGBoost

Every model runs inside a pipeline (`StandardScaler` plus an optional sampler) and is evaluated with **stratified 5-fold cross-validation**.

### 4. Handling class imbalance
Each model is trained in three variants:
- no resampling
- **SMOTE** (oversampling the minority class)
- **Random undersampling** of the majority class

Each variant is trained on three feature sets: all features, the 6 selected features and the 10 selected features. That gives **63 model configurations**, compared on accuracy, macro precision/recall/F1 and **per-class precision and recall**.

### 5. Hyperparameter optimization
`RandomizedSearchCV` (3-fold) on a SMOTE pipeline for all 7 models, using both reduced feature sets. Each tuned model is then evaluated on a held-out 20% stratified test set, with confusion matrices and minority-class metrics.

## Results

Cross-validated results on the full feature set, without resampling:

| Model               | Accuracy   | Macro F1  | Recall (safe) | Precision (safe) |
|---------------------|:----------:|:---------:|:-------------:|:----------------:|
| **XGBoost**         | **96.89%** | **0.919** | **0.806**     | **0.911**        |
| Random Forest       | 95.97%     | 0.888     | 0.698         | 0.931            |
| Decision Tree       | 95.16%     | 0.883     | 0.811         | 0.775            |
| SVM (RBF)           | 94.02%     | 0.822     | 0.547         | 0.885            |
| Logistic Regression | 90.73%     | 0.700     | 0.332         | 0.697            |
| KNN                 | 90.35%     | 0.697     | 0.342         | 0.645            |
| Naive Bayes         | 84.77%     | 0.692     | 0.600         | 0.391            |

**Key findings:**
- **XGBoost was the best model overall.** On the full feature set it had the highest macro F1 in all three variants: no resampling, SMOTE and undersampling.
- **Accuracy alone is misleading here.** Logistic Regression reaches 90.7% accuracy but finds only a third of the safe samples. With 88.6% of samples in one class, minority-class recall and macro F1 are the metrics that actually tell the models apart.
- **Resampling trades precision for recall.** Undersampling pushed XGBoost's recall on safe samples from 0.81 to **0.95**, while precision dropped from 0.91 to 0.62. SMOTE gave the most balanced result (recall 0.83, precision 0.85).
- **Feature selection cost accuracy.** Tree-based models lost 2–6 percentage points on the reduced sets. KNN was the only model that improved (from 76.4% to 82.0% with SMOTE), because it benefits from fewer, more relevant dimensions.

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · SciPy · Matplotlib · seaborn · Jupyter

## Running the notebook

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn xgboost matplotlib seaborn shap jupyter
jupyter notebook Masinsko1.ipynb
```
