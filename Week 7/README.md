# Clustering with EM Algorithm and k-Means

## Introduction
This application performs clustering analysis on the Iris dataset using the Expectation-Maximization (EM) algorithm with Gaussian Mixture Model (GMM) and the k-Means algorithm. It displays the clustering results and evaluates the performance using silhouette scores.

## Dataset
The Iris dataset is used for this analysis. It contains measurements of sepal and petal dimensions for three species of iris flowers: Setosa, Versicolour, and Virginica.

|   | Sepal Length (cm) | Sepal Width (cm) | Petal Length (cm) | Petal Width (cm) | Target |
|---|--------------------|------------------|-------------------|------------------|--------|
| 0 | 5.1                | 3.5              | 1.4               | 0.2              | 0      |
| 1 | 4.9                | 3.0              | 1.4               | 0.2              | 0      |
| 2 | 4.7                | 3.2              | 1.3               | 0.2              | 0      |
| 3 | 4.6                | 3.1              | 1.5               | 0.2              | 0      |
| 4 | 5.0                | 3.6              | 1.4               | 0.2              | 0      |

## Silhouette Scores
- **GMM Silhouette Score:** 0.3530
- **k-Means Silhouette Score:** 0.5528

## Clustering Results

### GMM Clustering Results

The scatter matrix shows the clustering results obtained by the Gaussian Mixture Model (GMM) algorithm. Each point represents an instance in the dataset, colored according to the assigned cluster. The shape of the point indicates the species of the iris flower.

### k-Means Clustering Results

Similarly, the scatter matrix displays the clustering results obtained by the k-Means algorithm. Each point represents an instance in the dataset, colored by the assigned cluster and shaped according to the species of the iris flower.

## Note
This application demonstrates clustering analysis using two different algorithms: Gaussian Mixture Model (GMM) and k-Means. The silhouette scores provide a measure of how well-defined the clusters are. For real-world applications, consider using more comprehensive and current data, and tuning the parameters of the clustering algorithms for better results.

## Links:
- [Streamlit App](https://info12.streamlit.app/)
- [Medium Article](https://medium.com/@vikramkaruppannan6807/em-algorithm-0a4b70533919)

