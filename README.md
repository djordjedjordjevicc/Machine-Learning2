# Student Social Media Addiction – Clustering Analysis

An unsupervised learning project that groups students by their social media habits, sleep, mental health and academic impact. It compares **9 clustering algorithms** and checks whether they find the same student profiles.

All work is in [`Untitled.ipynb`](Untitled.ipynb). Notebook comments are in Serbian.

## Dataset

[`Students Social Media Addiction.csv`](Students%20Social%20Media%20Addiction.csv) is a public survey dataset (Kaggle) of **705 students** from 110 countries, aged 18–24.

| Type        | Features                                                                                                               |
|-------------|------------------------------------------------------------------------------------------------------------------------|
| Numeric     | Age, Avg_Daily_Usage_Hours, Sleep_Hours_Per_Night, Mental_Health_Score, Conflicts_Over_Social_Media, Addicted_Score     |
| Categorical | Gender, Academic_Level, Country, Most_Used_Platform, Affects_Academic_Performance, Relationship_Status                  |

## Pipeline

### 1. Exploratory data analysis
- Distributions of all numeric and categorical attributes
- Correlation matrices before and after encoding
- Outlier detection with the IQR method (none found)

### 2. Preprocessing
Each categorical attribute gets the encoding that suits its cardinality and meaning:

| Encoding    | Attributes                                                        | Why                                     |
|-------------|-------------------------------------------------------------------|-----------------------------------------|
| One-hot     | Gender, Affects_Academic_Performance, Relationship_Status         | Few categories, no natural order        |
| Ordinal     | Academic_Level (High School < Undergraduate < Graduate)           | Ordered categories                      |
| Target      | Country (110 values), Most_Used_Platform (12 values)              | Too many categories for one-hot         |

All features are then standardized with `StandardScaler`.

### 3. Choosing the number of clusters
- **Elbow method** (SSE, with `kneed`) suggested k = 3
- **Silhouette coefficient** and **Calinski–Harabasz** (with `yellowbrick`) suggested k = 2
- **BIC / AIC** for Gaussian Mixture suggested k = 5
- **Dendrogram** (Ward linkage) for hierarchical clustering

### 4. Algorithms

| Algorithm                       | Configurations tested           |
|---------------------------------|---------------------------------|
| K-Means                         | k = 2, 3, 4                     |
| Agglomerative Clustering        | k = 2, 3                        |
| Hierarchical (Ward, SciPy)      | dendrogram cut at two heights   |
| Gaussian Mixture                | 5 components                    |
| Birch                           | k = 2, 3                        |
| Fuzzy C-Means (`scikit-fuzzy`)  | c = 2                           |
| Bisecting K-Means               | k = 2                           |
| MeanShift                       | estimated bandwidth             |
| OPTICS                          | density-based, with noise       |

### 5. Visualization and interpretation
- Clusters projected with **PCA** (2D and 3D; 6 components explain 92% of the variance) and **t-SNE**
- **Feature importance** per clustering, measured as the standard deviation of cluster centroids across features
- **Heatmaps** of the average scaled value of each attribute per cluster, used to describe each group of students

## Results

The silhouette and Calinski–Harabasz scores were highest at **k = 2** (silhouette 0.33 for K-Means). K-Means, Agglomerative, Birch, Fuzzy C-Means, Bisecting K-Means and the Ward dendrogram all produced **almost identical** two-cluster solutions, so the split is a stable structure in the data rather than an artifact of one algorithm.

Average standardized values in the two clusters (K-Means, k = 2):

| Attribute                      | Cluster A: *healthy usage* | Cluster B: *at risk* |
|--------------------------------|:--------------------------:|:--------------------:|
| Avg daily usage (hours)        | −0.88                      | +0.52                |
| Sleep per night (hours)        | +0.84                      | −0.50                |
| Mental health score            | +1.07                      | −0.63                |
| Conflicts over social media    | −1.07                      | +0.63                |
| Addicted score                 | −1.13                      | +0.67                |
| Affects academic performance   | −1.27                      | +0.75                |

**What this means:** students who spend more time on social media also sleep less, report worse mental health, have more conflicts over social media and more often say it hurts their studies. These attributes move together, so the main dividing line is overall usage intensity. Age, gender and relationship status play only a small role in the split.

Using more clusters (K-Means k = 3/4, Gaussian Mixture k = 5) mostly splits these two groups further by **age and academic level**.

## Tech stack

Python · pandas · NumPy · scikit-learn · SciPy · category_encoders · yellowbrick · kneed · scikit-fuzzy · Matplotlib · seaborn · Jupyter

## Running the notebook

```bash
pip install pandas numpy scipy scikit-learn category_encoders yellowbrick kneed scikit-fuzzy matplotlib seaborn joblib jupyter
jupyter notebook Untitled.ipynb
```

