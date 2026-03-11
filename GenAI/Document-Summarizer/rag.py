from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load models 
embedder = SentenceTransformer("all-MiniLM-L6-v2")

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

# Process PDF → Vector DB
def process_pdf(filepath):

    loader = PyPDFLoader(filepath)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.split_documents(pages)

    texts = [doc.page_content for doc in docs]

    from langchain_community.embeddings import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(texts, embedding_model)

    return vectorstore

# Ask question
def ask_question(vectorstore, question):

    docs = vectorstore.similarity_search(question, k=3)

    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer based only on the context:

    {context}

    Question: {question}
    """

    result = llm(prompt)

    return result[0]['generated_text']