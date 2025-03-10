import os
import streamlit as st
import pickle
import tempfile
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.chains import RetrievalQAWithSourcesChain, load_qa_chain, load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI

# OpenAI API Key
OPENAI_API_KEY = 'Your_OpenAI_API_KEY'
URL_STORE = "url_faiss_store.pkl"
PDF_STORE = "pdf_faiss_store.pkl"

# Streamlit Page Config
st.set_page_config(
    page_title="Personal AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- UI Enhancements ----
st.markdown(
    """
    <style>
        .big-font { font-size:30px !important; font-weight: bold; color: #4e73df; text-align: center; }
        .subheader-font { font-size:20px !important; color: #2d3e50; font-weight: bold; }
        .sidebar .sidebar-content { background-color: #f0f5ff; }
        .stButton>button { width: 100%; background-color: #4e73df; color: white; border-radius: 8px; padding: 10px; }
        .stButton>button:hover { background-color: #365f96; }
        .stTextInput>label { font-size: 16px; font-weight: bold; color: #333; }
        .stRadio>label { font-size: 16px; font-weight: bold; color: #333; }
        .stMarkdown { font-size: 16px; color: #333; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='big-font'>🤖 Personal AI Assistant</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 class='subheader-font'>🔹 Assistant Console</h2>", unsafe_allow_html=True)

# Load OpenAI LLM
llm = OpenAI(temperature=0.7, max_tokens=500, openai_api_key=OPENAI_API_KEY)

def load_or_create_vectorstore(store_path, docs):
    if os.path.exists(store_path):
        with open(store_path, "rb") as f:
            return pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(store_path, "wb") as f:
            pickle.dump(vectorstore, f)
        return vectorstore

# ---- Sidebar: URL Input ----
with st.sidebar.expander("🌐 Process URLs", expanded=True):
    num_links = st.slider("Number of URLs:", 1, 5, 1)
    urls = [st.text_input(f"URL {i+1}", key=f"url{i}") for i in range(num_links)]

url_vectorstore = None
if any(urls):
    loader = UnstructuredURLLoader(urls=urls)
    url_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    url_docs = text_splitter.split_documents(url_docs)
    if url_docs:
        url_vectorstore = load_or_create_vectorstore(URL_STORE, url_docs)

# ---- Sidebar: PDF Upload ----
with st.sidebar.expander("📄 Upload a PDF", expanded=True):
    uploaded_file = st.file_uploader("Choose a PDF", type=['pdf'])

pdf_vectorstore = None
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    pdf_loader = PyPDFLoader(temp_path)
    pdf_docs = pdf_loader.load_and_split()

    if pdf_docs:
        pdf_vectorstore = load_or_create_vectorstore(PDF_STORE, pdf_docs)

    os.remove(temp_path)

# ---- Query Section ----
st.subheader("💬 Ask a Question", anchor="query_section")

data_source = st.radio("Select Source:", ["URL", "PDF", "Both"], horizontal=True)
query = st.text_input("Enter your question:")

if query:
    results = []
    if data_source in ["URL", "Both"] and url_vectorstore:
        chain = RetrievalQAWithSourcesChain.from_llm(llm, retriever=url_vectorstore.as_retriever())
        url_result = chain({"question": query}, return_only_outputs=True)
        results.append(f"📌 **URL Answer:**\n{url_result['answer']}\n")
    
    if data_source in ["PDF", "Both"] and pdf_vectorstore:
        docs = pdf_vectorstore.similarity_search(query)
        chain = load_qa_chain(llm, chain_type="stuff")
        pdf_result = chain.run(input_documents=docs, question=query)
        results.append(f"📄 **PDF Answer:**\n{pdf_result}\n")
    
    if results:
        st.success("Here are your answers:")
        st.markdown("\n\n".join(results))
    else:
        st.warning("No data found for the selected source.")

# ---- Summarization Section ----
with st.expander("📜 Summarize PDF", expanded=True):
    if st.button("Summarize PDF") and pdf_vectorstore:
        with st.spinner("Summarizing PDF... ⏳"):
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(pdf_docs)
            st.markdown(f"### 📄 PDF Summary:\n{summary}")
