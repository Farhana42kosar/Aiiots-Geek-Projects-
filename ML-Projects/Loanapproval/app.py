import streamlit as st
import pandas as pd
import joblib


# Load model

model_path = "loan_logistic_model.pkl"  # replace with your Logistic model if needed
model = joblib.load(model_path)


st.title("Loan Approval Prediction")
st.write("Enter applicant details below to predict loan approval:")

# Input form
with st.form("loan_form"):
    # Applicant info
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Income", min_value=0, value=50000)
    person_emp_exp = st.number_input("Years of Employment", min_value=0, value=5)
    
    # Loan info
    loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0, step=0.1)
    loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    
    # Credit info
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    
    # Categorical
    person_gender_male = st.selectbox("Gender", ["Male", "Female"])
    person_gender_male = 1 if person_gender_male=="Male" else 0
    
    education_options = ["Bachelor", "Doctorate", "High School", "Master"]
    person_education = st.selectbox("Education", education_options)
    
    home_options = ["OWN", "RENT", "OTHER"]
    person_home_ownership = st.selectbox("Home Ownership", home_options)
    
    loan_intent_options = ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
    loan_intent = st.selectbox("Loan Intent", loan_intent_options)
    
    previous_loan_defaults = st.selectbox("Previous Loan Defaults?", ["Yes", "No"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Loan Approval")
    
    if submitted:
        
        #  input dataframe
        
        sample_input = {
            'person_age': person_age,
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score,
            'person_gender_male': person_gender_male,
            'person_education_Bachelor': 1 if person_education=="Bachelor" else 0,
            'person_education_Doctorate': 1 if person_education=="Doctorate" else 0,
            'person_education_High School': 1 if person_education=="High School" else 0,
            'person_education_Master': 1 if person_education=="Master" else 0,
            'person_home_ownership_OTHER': 1 if person_home_ownership=="OTHER" else 0,
            'person_home_ownership_OWN': 1 if person_home_ownership=="OWN" else 0,
            'person_home_ownership_RENT': 1 if person_home_ownership=="RENT" else 0,
            'loan_intent_EDUCATION': 1 if loan_intent=="EDUCATION" else 0,
            'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent=="HOMEIMPROVEMENT" else 0,
            'loan_intent_MEDICAL': 1 if loan_intent=="MEDICAL" else 0,
            'loan_intent_PERSONAL': 1 if loan_intent=="PERSONAL" else 0,
            'loan_intent_VENTURE': 1 if loan_intent=="VENTURE" else 0,
            'previous_loan_defaults_on_file_Yes': 1 if previous_loan_defaults=="Yes" else 0
        }
        
        input_df = pd.DataFrame([sample_input])
        input_df = input_df[model.feature_names_in_]  # ensure column order matches training
        
        #prediction
        prob_default = model.predict_proba(input_df)[0][1]
        approval_prob = 1 - prob_default
        threshold = 0.6
        
        decision = "Loan Approved" if approval_prob >= threshold else "Loan Rejected"
        
        
        st.success(f"Decision: {decision}")
        st.info(f"Approval Probability: {round(approval_prob,3)}")
