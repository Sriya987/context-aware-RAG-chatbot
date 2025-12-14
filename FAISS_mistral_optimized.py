import streamlit as st
import os
from pypdf import PdfReader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)


@st.cache_resource
def load_faiss_vector_store(path="faiss_index"):
    embeddings = get_embeddings()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def get_llm():
    return Ollama(model="neural-chat:latest",  # Much faster than mistral
        temperature=0.3,
        num_predict=200,
        num_thread=12,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.1)


def build_qa_chain():
    vector_store = load_faiss_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = PromptTemplate(
        template="""
Use the following context to answer the question.
If you cannot answer from the context, say:
"I don't have enough information."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )



st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")

st.title("📄 RAG Chatbot (PDF + FAISS + Ollama)")
st.write("Upload a PDF once and ask **unlimited questions**.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    try:
        if "qa" not in st.session_state:
            os.makedirs("uploaded", exist_ok=True)
            pdf_path = os.path.join("uploaded", uploaded_file.name)

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("📄 Extracting text..."):
                text = extract_text_from_pdf(pdf_path)

            if not os.path.exists("faiss_index"):
                with st.spinner("🔄 Creating vector store..."):
                    create_faiss_vector_store(text)

            with st.spinner("🤖 Loading QA chain..."):
                st.session_state.qa = build_qa_chain()

            st.success("PDF indexed successfully!")

        query = st.text_input("Ask a question from the PDF:")

        ask = st.button("Ask")

        if ask and query:
            with st.spinner("🔍 Generating answer..."):
                answer = st.session_state.qa.invoke(query)

            st.markdown("### 📌 Answer")
            st.write(answer)


    except Exception as e:
        st.error("❌ App crashed")
        st.exception(e)
