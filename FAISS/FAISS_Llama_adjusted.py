
import streamlit as st
import os
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    @st.cache_resource(show_spinner=False)
    def get_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)

@st.cache_resource(show_spinner=False)
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

@st.cache_resource(show_spinner=False)
def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    @st.cache_resource(show_spinner=False)
    def get_llm():
        return Ollama(model="llama3.2")
    llm = get_llm()

    template = """Use the following context to answer the question. If you cannot answer based on the context, say "I don't have enough information."

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

st.title("RAG Chatbot with FAISS and LLaMA")
st.write("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = f"uploaded/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("📄 Extracting text from PDF..."):
        text = extract_text_from_pdf(pdf_path)
    st.success(f"✅ Extracted {len(text)} characters from PDF")

    with st.spinner("🔄 Splitting text into chunks..."):
        create_faiss_vector_store(text)
    st.success("✅ Vector store created successfully!")

    with st.spinner("🤖 Loading QA chain..."):
        qa = build_qa_chain()
    st.success("✅ Ready to answer questions!")

    query = st.text_input("Ask something from the PDF:")

    if query:
        with st.spinner("🔍 Searching and generating answer..."):
            response = qa.invoke(query)
        st.write("### 📌 Answer:")
        st.write(response)