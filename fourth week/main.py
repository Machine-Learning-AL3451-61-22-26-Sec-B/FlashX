import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import graphviz

# Function to load data
@st.cache
def load_data(file):
    return pd.read_csv(file)

# Function to preprocess data and build models
def preprocess_and_build_models(df):
    # Data preprocessing
    X = df.drop('fraud', axis=1)
    Y = df['fraud']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Decision Tree classifier
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
    clf.fit(X_train, Y_train)
    Y_pred_dt = clf.predict(X_test)
    dt_accuracy = accuracy_score(Y_test, Y_pred_dt)

    # Naive Bayes classifier
    clf_nb = GaussianNB()
    clf_nb.fit(X_train, Y_train)
    Y_pred_nb = clf_nb.predict(X_test)
    nb_accuracy = accuracy_score(Y_test, Y_pred_nb)

    return dt_accuracy, nb_accuracy, clf

# Main function
def main():
    # Title and description
    st.title("Fraud Detection Streamlit App")
    st.write("This app allows you to upload your own dataset for fraud detection.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        # Data exploration
        st.write("Data Overview:")
        st.write(df.head())
        st.write("Data Info:")
        st.write(df.info())
        st.write("Missing Values:")
        st.write(df.isna().sum())

        # Data visualization
        # Add your visualization code here

        # Model building and evaluation
        dt_accuracy, nb_accuracy, clf = preprocess_and_build_models(df)

        # Comparison
        st.write("Model Comparison:")
        st.write("Decision Tree Accuracy:", dt_accuracy)
        st.write("Naive Bayes Accuracy:", nb_accuracy)

        # Bar chart
        st.write("Accuracy Comparison Chart:")
        plt.bar(["Decision Tree", "Naive Bayes"], [dt_accuracy, nb_accuracy])
        plt.ylabel("Accuracy")
        st.pyplot()

        # Decision Tree visualization
        st.write("Decision Tree Visualization:")
        dot_data = export_graphviz(clf, out_file=None, feature_names=df.columns[:-1], filled=True)
        graph = graphviz.Source(dot_data)
        st.graphviz_chart(graph)

if __name__ == "__main__":
    main()
