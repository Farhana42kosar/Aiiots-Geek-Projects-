import streamlit as st
import requests
import base64

st.set_page_config(page_title="Resume Analyzer", layout="wide")

# CUSTOM CSS

st.markdown("""
    <style>
        /* Center the main title */
        .main-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 2rem;
        }
        /* Bigger Target Domain */
        .target-domain {
            font-size: 1.5rem;
            font-weight: 600;
            color: #4338CA;
            background-color: #EEF2FF;
            padding: 8px 15px;
            border-radius: 6px;
            display: inline-block;
            margin-bottom: 20px;
        }
        /* Bigger Matching Roles */
        .matching-job {
            font-size: 1.2rem;
            margin-bottom: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# SESSION STATE INITIALIZATION
if "data" not in st.session_state:
    st.session_state.data = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# HEADER
st.markdown('<h1 class="main-title">Resume Analyzer</h1>', unsafe_allow_html=True)

# FUNCTIONS
def reset_app():
    st.session_state.data = None
    st.session_state.uploaded_file = None

# SCREEN 1 — UPLOAD RESUME
if st.session_state.data is None:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("Upload your resume to see your score")
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF)", type=["pdf"], label_visibility="hidden"
        )

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            if st.button("Analyze Resume"):
                with st.spinner("Processing..."):
                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/predict",
                            files={"file": uploaded_file}
                        )
                        response.raise_for_status()
                        st.session_state.data = response.json()
                    except requests.exceptions.HTTPError as http_err:
                        st.error(f"HTTP error: {http_err} ({response.status_code})")
                    except requests.exceptions.ConnectionError:
                        st.error("Connection failed. Check your API.")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")


# SCREEN 2 — RESULTS

else:
    data = st.session_state.data
    ats_score = int(data["ats_score"])
    predicted_domain = data["predicted_domain"]
    job_recommendations = data.get("job_recommendations", [])

    col_left, col_right = st.columns([2, 3], gap="large")

    # ----- LEFT PANEL -----
    with col_left:
        st.subheader("ATS Score")
        st.metric(label="", value=f"{ats_score}/100")

        st.markdown("---")
        st.subheader("Predicted Domain")
        st.markdown(f'<div class="target-domain">{predicted_domain}</div>', unsafe_allow_html=True)

        st.subheader("Recommended Job Roles")
        for job in job_recommendations:
            st.markdown(f'<div class="matching-job">- {job["job"]} ({job["match"]}% match)</div>', unsafe_allow_html=True)

        st.button("Analyze Another Resume", on_click=reset_app)

    # ----- RIGHT PANEL -----
    with col_right:
        st.subheader("Resume Preview")
        uploaded_file = st.session_state.uploaded_file
        uploaded_file.seek(0)
        base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
        st.markdown(f'''
            <iframe src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" height="800px" style="border:none;"></iframe>
        ''', unsafe_allow_html=True)
