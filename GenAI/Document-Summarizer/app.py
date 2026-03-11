import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import base64
import os
import torch
torch.set_num_threads(2)
from rag import process_pdf, ask_question

#  Page Config
# -------------------------------
st.set_page_config(page_title="AI PDF Assistant", layout="wide")

 
# Custom UI Styling
st.markdown("""
    <style>
    /* Chat container */
    .chat-box {
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        font-size: 16px;
        line-height: 1.5;
    }

    /* User message */
    .user-msg {
        background-color: #2563eb;  /* blue */
        color: white;
        text-align: right;
    }

    /* Bot message */
    .bot-msg {
        background-color: #f1f5f9;  /* light gray */
        color: black;
        text-align: left;
    }

    /* Improve overall text visibility */
    body {
        color: #111827;
    }
    </style>
""", unsafe_allow_html=True)

#  Load summarization model 
import os

if not os.path.exists("data"):
    os.makedirs("data")
@st.cache_resource
def load_pipeline():
    checkpoint = "MBZUAI/LaMini-Flan-T5-248M"

    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

    pipe = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer
    )
    return pipe

pipe_sum = load_pipeline()


#  Display PDF

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f'''
    <iframe src="data:application/pdf;base64,{base64_pdf}" 
    width="100%" height="600"></iframe>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)

#  Summarization 

def llm_pipeline(filepath):

    loader = PyPDFLoader(filepath)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(pages)

    summaries = []
    progress = st.progress(0)

    for i, doc in enumerate(docs):
        chunk = doc.page_content

        try:
            result = pipe_sum(
                chunk,
                max_length=120,
                min_length=30,
                do_sample=False
            )

            summary = result[0]['summary_text']
            summaries.append(summary)

        except Exception as e:
            st.error(f"Error: {e}")
            continue

        progress.progress((i + 1) / len(docs))

    final_summary = " ".join(summaries)

    final_summary = pipe_sum(
        final_summary,
        max_length=150,
        min_length=50,
        do_sample=False
    )[0]['summary_text']

    return final_summary

#  MAIN APP
def main():

    st.markdown("<div class='title'>📄 AI PDF Assistant</div>", unsafe_allow_html=True)

    option = st.radio("Choose Feature", ["Summarize PDF", "Chat with PDF"])

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:

        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", uploaded_file.name)

        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())

        # -------------------------------
        # 📄 SUMMARIZATION
        # -------------------------------
        if option == "Summarize PDF":

            if st.button("🚀 Summarize"):

                col1, col2 = st.columns(2)

                with col1:
                    st.info("📂 Uploaded File")
                    displayPDF(filepath)

                with col2:
                    with st.spinner("⏳ Summarizing..."):
                        summary = llm_pipeline(filepath)

                    st.success("✅ Done")
                    st.write(summary)

        #  CHAT WITH PDF
        elif option == "Chat with PDF":

            st.subheader("💬 Chat with your PDF")

            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Create vectorstore once
            if "vectorstore" not in st.session_state:
                with st.spinner("📚 Processing PDF..."):
                    st.session_state.vectorstore = process_pdf(filepath)
                st.success("✅ PDF Ready!")

            # User input
            user_input = st.text_input("Ask something...")

            if user_input:

                answer = ask_question(st.session_state.vectorstore, user_input)

                # Store chat
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", answer))

            # Display chat
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f'<div class="chat-box user-msg">🧑 {msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-box bot-msg">🤖 {msg}</div>', unsafe_allow_html=True)

# -------------------------------
if __name__ == "__main__":
    main()