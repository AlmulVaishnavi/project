# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Title for the app
st.title("AI-Driven Anomaly Detection in Smart Manufacturing")

# Description
st.write("""
This application allows you to perform anomaly detection on the manufacturing 6G dataset.
You can visualize the data, train a model, and explore detected anomalies.
""")

# Step 1: Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('manufacturing_6G_dataset.csv')
    return data

df = load_data()

# Show the first few rows of the dataset
st.write("Dataset Preview:", df.head())

# Step 2: Data Preprocessing
st.subheader("Data Preprocessing")
# Check for missing values and handle them
if df.isnull().sum().any():
    st.warning("Dataset contains missing values!")
    st.write(df.isnull().sum())
    df = df.dropna()  # Removing rows with missing data (you can also use imputation)
else:
    st.success("No missing values in the dataset.")

# Standardize the features (important for many ML models)
st.write("Standardizing the data...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))  # Scale only numeric columns

# Step 3: Anomaly Detection using Isolation Forest
st.subheader("Anomaly Detection")
st.write("Training the anomaly detection model using Isolation Forest...")

# Initialize IsolationForest model (use n_estimators and contamination as per your dataset's characteristics)
model = IsolationForest(n_estimators=100, contamination=0.05)  # 5% outliers (adjust as needed)
model.fit(scaled_data)

# Predict anomalies (-1 indicates anomaly, 1 indicates normal)
predictions = model.predict(scaled_data)

# Add predictions to the dataframe
df['anomaly'] = predictions

# Step 4: Visualizing the results
st.subheader("Anomaly Detection Results")

# Show number of anomalies and normal points
st.write(f"Number of Anomalies Detected: {np.sum(df['anomaly'] == -1)}")
st.write(f"Number of Normal Points: {np.sum(df['anomaly'] == 1)}")

# Show a scatter plot (if there are two features, otherwise this is for demonstration)
if df.shape[1] >= 2:
    st.write("Displaying Anomalies vs Normal Points in a scatter plot")
    fig, ax = plt.subplots()
    ax.scatter(df[df['anomaly'] == 1].iloc[:, 0], df[df['anomaly'] == 1].iloc[:, 1], color='g', label='Normal')
    ax.scatter(df[df['anomaly'] == -1].iloc[:, 0], df[df['anomaly'] == -1].iloc[:, 1], color='r', label='Anomaly')
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Not enough features for a scatter plot. Showing a heatmap instead.")
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot()

# Step 5: Allow users to interact with the results
st.subheader("Explore Anomalies")
st.write("Explore specific anomalies detected by the model:")
anomalies = df[df['anomaly'] == -1]
st.write(anomalies)

# Step 6: Additional functionality (optional)
# Allow the user to download the dataset with anomalies labeled
csv = df.to_csv(index=False)
st.download_button("Download Dataset with Anomalies", csv, "anomalies_detected.csv", "text/csv")

