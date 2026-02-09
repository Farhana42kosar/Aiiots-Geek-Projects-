import streamlit as st
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit page setup
st.set_page_config(
    page_title="🎬 Movie Review Sentiment Analyzer",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.5em;
        color: #FF4B4B;
        font-weight: 700;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2em;
        color: #f0f0f0;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        background-color: #1E1E1E !important;
        color: #ffffff !important;
        border-radius: 10px;
        font-size: 1em;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ffb347);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #ffb347, #ff4b4b);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title section
st.markdown("<h1 class='main-title'>🎞️ Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter a movie review below and find out whether it’s Positive or Negative!</p>", unsafe_allow_html=True)

# Input text box
review = st.text_area("📝 Write your movie review here:", height=150, placeholder="e.g., I loved the acting and storyline!")

# Predict button
if st.button("🎯 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        # Transform and predict
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]

        # Show result
        if prediction == 1:
            st.success(" **Positive Review!** This movie seems to be a hit! ")
        else:
            st.error("**Negative Review!** Looks like this movie didn’t impress much. ")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Developed by Farhana 🎬</p>", unsafe_allow_html=True)
