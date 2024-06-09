import streamlit as st
import numpy as np
import pandas as pd

# Define the learn function using the Candidate Elimination algorithm
def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    indices = [i for i, val in enumerate(general_h) if val == ['?'] * len(specific_h)]
    for i in indices:
        general_h.remove(['?'] * len(specific_h))

    return specific_h, general_h

# Create the main function for the Streamlit app
def main():
    st.title('Candidate Elimination Algorithm App(created by TEAM FLASH X)')

    # Upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("## Dataset")
        st.write(data)

        concepts = np.array(data.iloc[:, 0:-1])
        target = np.array(data.iloc[:, -1])

        s_final, g_final = learn(concepts, target)
        st.write("\nFinal Specific_h:")
        st.write(s_final)
        st.write("\nFinal General_h:")
        st.write(g_final)

if __name__ == "__main__":
    main()
