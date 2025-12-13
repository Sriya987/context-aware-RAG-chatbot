import os
import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

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


def create_vector_store(text):
    split = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = split.split_text(text)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_texts(chunks, embedding=emb)
    store.save_local("faiss_index")


def load_vector_store():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)


def load_llm():
    model_id = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, max_new_tokens=200, do_sample=False)
    return HuggingFacePipeline(pipeline=gen)

def build_chain(store):
    retriever = store.as_retriever(search_kwargs={"k": 2})
    llm = load_llm()
    prompt = PromptTemplate.from_template(
        "Use the context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


st.title("📘 PDF RAG Chatbot (FLAN-T5 Offline)")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = f"uploaded/{uploaded.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Reading PDF...")
    text = extract_text_from_pdf(pdf_path)

    st.info("Creating vector store...")
    create_vector_store(text)
    store = load_vector_store()
    chain = build_chain(store)

    question = st.text_input("Ask something from the PDF")

    if question:
        with st.spinner("Generating answer..."):
            answer = chain.invoke(question)
        st.write("### Answer:")
        st.write(answer)
