import numpy as np
import pandas as pd
from hmmlearn.hmm import MultinomialHMM
import matplotlib.pyplot as plt
import streamlit as st

# Load the preprocessed data
data_cleaned = pd.read_csv('data_cleaned.csv')

st.title("Gene Identification Using HMM and Sequence Modelling")
st.subheader("Prediction of Coding and Non-Coding Regions")

# Ensure that 'genotype_numeric' column exists and the data is valid
data_cleaned = data_cleaned.dropna(subset=['genotype_numeric'])
data_cleaned = data_cleaned[data_cleaned['genotype_numeric'] >= 0]

# If the 'predicted_states' column doesn't exist, we need to generate it using HMM

if 'predicted_states' not in data_cleaned.columns:
    # Convert 'genotype_numeric' into observations for the HMM
    observations = data_cleaned['genotype_numeric'].values.reshape(-1, 1)

    # Initialize the HMM model
    hmm_model = MultinomialHMM(n_components=2, n_iter=200,random_state=122)

    # Fit the model
    hmm_model.fit(observations)

    # Predict hidden states
    predicted_states = hmm_model.predict(observations)

    # Add the predicted states back to the dataset
    data_cleaned['predicted_states'] = predicted_states

# Define coding and non-coding based on predicted_states
data_cleaned['region_type'] = np.where(data_cleaned['predicted_states'] == 1, 'Coding', 'Non-Coding')

# Plot coding and non-coding regions using different colors
fig, ax = plt.subplots(figsize=(10, 6))

# Plot coding regions (e.g., predicted_states == 1)
coding = data_cleaned[data_cleaned['region_type'] == 'Coding']
non_coding = data_cleaned[data_cleaned['region_type'] == 'Non-Coding']

# Scatter plot with different colors
ax.scatter(coding['position'], coding['predicted_states'], c='blue', label='Coding', s=10)
ax.scatter(non_coding['position'], non_coding['predicted_states'], c='red', label='Non-Coding', s=10)

# Labels and titles
ax.set_xlabel('Position')
ax.set_ylabel('Predicted States')
ax.set_title('HMM Predicted Coding and Non-Coding Regions')
ax.legend()

# Show the plot in Streamlit
st.pyplot(fig)

# Display the first few rows of the data
st.write(data_cleaned.head())

# Display column names
st.write(data_cleaned.columns)

# Display data in tabular format, focusing on the coding and non-coding regions
st.subheader("Predicted States Data with Coding/Non-Coding Regions")
st.write(data_cleaned[['position', 'chromosome', 'genotype', 'predicted_states', 'region_type']].head(100))
