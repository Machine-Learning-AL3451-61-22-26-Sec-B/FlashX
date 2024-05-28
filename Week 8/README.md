# k-Nearest Neighbors Classifier for Iris Dataset

## Introduction
This application implements a k-Nearest Neighbors (kNN) classifier for the Iris dataset. It allows users to adjust the number of neighbors and displays model performance metrics, including accuracy and a classification report.

## Dataset
The Iris dataset is used for this classification task. It contains measurements of sepal and petal dimensions for three species of iris flowers: Setosa, Versicolour, and Virginica.

|   | Sepal Length (cm) | Sepal Width (cm) | Petal Length (cm) | Petal Width (cm) | Target |
|---|--------------------|------------------|-------------------|------------------|--------|
| 0 | 5.1                | 3.5              | 1.4               | 0.2              | 0      |
| 1 | 4.9                | 3.0              | 1.4               | 0.2              | 0      |
| 2 | 4.7                | 3.2              | 1.3               | 0.2              | 0      |
| 3 | 4.6                | 3.1              | 1.5               | 0.2              | 0      |
| 4 | 5.0                | 3.6              | 1.4               | 0.2              | 0      |

## Model Performance
- **Accuracy:** 0.9778

## Classification Report

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Setosa        | 1.00      | 1.00   | 1.00     | 14      |
| Versicolour   | 1.00      | 0.94   | 0.97     | 16      |
| Virginica     | 0.93      | 1.00   | 0.96     | 15      |
|               |           |        |          |         |
| Accuracy      |           |        | 0.98     | 45      |
| Macro Avg     | 0.98      | 0.98   | 0.98     | 45      |
| Weighted Avg  | 0.98      | 0.98   | 0.98     | 45      |

## Sample Predictions

### Correct Predictions
- Features: [6.1, 2.8, 4.7, 1.2], True Label: Versicolour, Predicted Label: Versicolour
- Features: [6.0, 2.9, 4.5, 1.5], True Label: Versicolour, Predicted Label: Versicolour
- Features: [5.7, 2.6, 3.5, 1.0], True Label: Versicolour, Predicted Label: Versicolour

### Wrong Predictions
- Features: [5.4, 3.9, 1.7, 0.4], True Label: Setosa, Predicted Label: Versicolour
- Features: [5.0, 3.4, 1.5, 0.2], True Label: Setosa, Predicted Label: Versicolour

## Links:
- [Streamlit App](https://infosquad8.streamlit.app/)
- [Medium Article](https://medium.com/@pravinelango66/k-nearest-neighbour-algorithm-dbce89f4d79a)

