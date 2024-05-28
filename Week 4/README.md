# Credit Card Fraud Detection

## Overview

This Streamlit web application performs credit card fraud detection using two machine learning algorithms: Decision Tree and Naive Bayes. It allows users to upload a CSV file containing credit card transaction data and provides insights into the dataset, such as missing values, feature analysis, and model performance evaluation.

### Libraries Used

- Pandas: For data manipulation
- Matplotlib and Seaborn: For data visualization
- Scikit-learn: For machine learning algorithms and evaluation metrics
- Streamlit: For creating the web application interface
- Missingno: For visualizing missing values in the dataset
- Graphviz: For visualizing decision tree models

## Functionality

### Upload and Configure

Users can upload a CSV file containing credit card transaction data via the sidebar. They can then select the machine learning algorithm (Decision Tree or Naive Bayes) to perform fraud detection.

### Dataset Preview

The application displays the first five rows of the dataset to provide users with a preview of the data.

### Dataset Information

Information about the dataset, including the number of rows, columns, and data types, is displayed.

### Missing Values

The application identifies and displays missing values in the dataset, along with a visualization of the missing values using a bar chart.

### Feature Analysis

Users can explore the distribution of features in the dataset using boxplots.

### Model Performance

#### Decision Tree

The application trains a Decision Tree classifier on the dataset and evaluates its performance using a confusion matrix and accuracy score. It also visualizes the decision tree model.

#### Naive Bayes

Similarly, the application trains a Naive Bayes classifier and evaluates its performance using a confusion matrix and accuracy score.

### Accuracy Comparison

The application compares the accuracy of the Decision Tree and Naive Bayes classifiers using a bar chart.

## Usage

To use the application, follow these steps:

1. Upload a CSV file containing credit card transaction data.
2. Select the machine learning algorithm (Decision Tree or Naive Bayes) to perform fraud detection.
3. Explore dataset insights, such as missing values and feature analysis.
4. Evaluate model performance using confusion matrices and accuracy scores.
5. Compare the accuracy of different classifiers using a bar chart.

## Conclusion

This Streamlit web application provides a user-friendly interface for credit card fraud detection, allowing users to analyze datasets, train machine learning models, and evaluate model performance.

## Links:
- [Streamlit App]()
- [Medium Article]()
