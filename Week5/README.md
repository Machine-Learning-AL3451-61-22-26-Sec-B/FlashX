# Naive Bayes Classifier for IMDb Review Classification

## Introduction
This application is a Naive Bayes classifier designed to predict whether an IMDb review is positive or negative based on its sentiment. It utilizes the Multinomial Naive Bayes algorithm for classification.

## Getting Started
To run the application locally, follow these steps:

1. Ensure you have Python installed on your system.
2. Install the required libraries by running:
    ```
    pip install pandas streamlit scikit-learn
    ```
3. Clone the repository or download the Python script (`app.py`).
4. Run the script using Streamlit by executing:
    ```
    streamlit run app.py
    ```

## Data Source
The dataset used for training and testing the classifier is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) from the UCI Machine Learning Repository. The dataset contains labeled SMS messages, where each message is labeled as 'ham' (not spam) or 'spam'.

## Code Overview
The code consists of the following main components:

1. **Downloading and Preparing Data**: The script downloads the dataset from the provided URL, extracts it, and prepares it for training and testing.

2. **Training the Classifier**: It splits the dataset into training and testing sets, initializes a CountVectorizer to convert text data into numerical feature vectors, and trains a Multinomial Naive Bayes classifier.

3. **Evaluation**: The trained classifier is used to predict the sentiment of the test set. Performance metrics such as confusion matrix, accuracy, precision, and recall are calculated and displayed.

## Usage
Once the application is running, it provides a user interface where users can interact with the classifier. Users can see the first five rows of the dataset, dataset information, missing values, confusion matrix, accuracy score, precision, and recall.

## Contributing
Contributions are welcome! If you have suggestions or find any issues, feel free to open an issue or submit a pull request.


## Links:
- [Streamlit App](https://teamflashx-naivebayesclassifier.streamlit.app/)
- [Medium Article](https://medium.com/@suryanehru0401/title-unveiling-sentiment-analysis-with-naive-bayes-classifier-23945aa7db80)
