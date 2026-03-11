import streamlit as st
import pandas as pd
import joblib

# Load model artifacts
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction")

st.write("Enter house details to predict price")

# ----------- USER INPUTS -------------
gr_liv_area = st.number_input("Living Area (sqft)", 300, 5000, 1500)
total_bsmt_sf = st.number_input("Total Basement Area (sqft)", 0, 3000, 800)
lot_area = st.number_input("Lot Area (sqft)", 1000, 20000, 7000)

overall_qual = st.slider("Overall Quality (1 = Poor, 10 = Excellent)", 1, 10, 7)
overall_cond = st.slider("Overall Condition (1 = Poor, 9 = Excellent)", 1, 9, 6)

year_built = st.number_input("Year Built", 1870, 2025, 2005)
year_remod = st.number_input("Year Remodeled", 1870, 2025, 2010)

garage_cars = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3])
full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3])
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])

neighborhood = st.selectbox(
    "Neighborhood",
    ["NridgHt", "NoRidge", "StoneBr", "Somerst", "CollgCr", "Crawfor", "GrnHill", "Timber", "Veenker", "Other"]
)

# ----------- BUILD INPUT DATA -----------

input_data = {
    'Gr Liv Area': gr_liv_area,
    'Total Bsmt SF': total_bsmt_sf,
    'Lot Area': lot_area,
    'Overall Qual': overall_qual,
    'Overall Cond': overall_cond,
    'Year Built': year_built,
    'Year Remod/Add': year_remod,
    'Garage Cars': garage_cars,
    'Full Bath': full_bath,
    'Bedroom AbvGr': bedrooms,
    'Neighborhood_Simple': neighborhood
}

input_df = pd.DataFrame([input_data])

# One-hot encode neighborhood
input_df = pd.get_dummies(input_df, columns=['Neighborhood_Simple'])

# Add missing columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure correct order
input_df = input_df[feature_columns]

# ----------- PREDICT -----------

if st.button("Predict Price"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    st.success(f"Estimated House Price: ₹ {prediction[0]:,.0f}")
