# app.py
import streamlit as st
import pickle
import numpy as np
import joblib
st.title("Customer Segmentation Predictor")

# Step 1: Load pre-trained models directly from files
with open("scaler.pkl", "rb") as f:
    scaler = joblib.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = joblib.load(f)

with open("cluster_names.pkl", "rb") as f:
    cluster_names = joblib.load(f)  # e.g., ['Low Value', 'Medium', 'High']

st.success("Models loaded successfully!")

# Step 2: Input customer data in the frontend
st.header("Enter Customer Details")

# Example: features your model expects
income = st.number_input("Annual Income (in $)", min_value=0, value=50000)
spending_score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=50)

input_data = np.array([[ income, spending_score]])

# Step 3: Scale and predict cluster
scaled_data = scaler.transform(input_data)
cluster_label = kmeans.predict(scaled_data)[0]
cluster_name = cluster_names[cluster_label]

# Step 4: Show result
st.subheader("Predicted Customer Segment")
st.write(f"Cluster: {cluster_label} → **{cluster_name}**")
