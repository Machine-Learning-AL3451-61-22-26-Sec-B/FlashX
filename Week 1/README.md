# Candidate Elimination Algorithm

## Overview:

This program implements the Candidate Elimination Algorithm, a concept of learning algorithm in Machine Learning. It provides both specific and general hypotheses based on the input data provided as a CSV file.

## Installation:

To run the Python code, make sure Python is installed and import the required libraries:

- NumPy
- Pandas
- Streamlit

## Execution:

Execute the Python file `app.py` and pass the path of the dataset as an argument:

```bash
python app.py trainingdata.csv
```

## Functionality:

The script performs the following tasks:

- Initializes specific and general hypotheses.
- Iterates through the algorithm steps.
- Prints the final specific and general hypotheses.

## Algorithm Steps:

1. Initialize the specific hypothesis with the first instance in the dataset.
2. Initialize the general hypothesis with all attributes set to "?".
3. Iterate through each instance in the dataset:
   - If the target concept is "yes", update the specific and general hypotheses accordingly.
   - If the target concept is "no", update the general hypothesis accordingly.
4. Remove any redundant hypotheses from the final general hypothesis.

## Links:

- [Streamlit Demo]()
- [Medium Article]()