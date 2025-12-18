import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
st.set_page_config(page_title="FAISS Neural-Chat RAG", layout="wide")

prompt = PromptTemplate(
    template="""Answer briefly based only on the context.
If the context is insufficient, say "I don't have enough information".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
)


@st.cache_resource
def get_llm():
    return OllamaLLM(
        model="llama3.2",
        temperature=0.3,
        base_url="http://127.0.0.1:11434",
        num_predict=256,
        k=5,
    )

@st.cache_resource
def get_embeddings():
    if "embeddings" not in st.session_state:
        with st.spinner("Loading embedding model (one-time)..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return st.session_state.embeddings


def load_documents(file_paths):
    texts = []
    for path in file_paths:
        reader = PdfReader(path)
        content = ""
        for p in reader.pages:
            content += p.extract_text() or ""
        texts.append(content)
    return "\n\n".join(texts)


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=30,
    )
    chunks = splitter.split_text(text)
    st.info(f"📊 Created {len(chunks)} chunks")
    return chunks


@st.cache_data(show_spinner=False)
def build_vector_store(chunks):
    embeddings = get_embeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)


# --------------------------------------------------
# RAG CHAIN (BUILT ONCE)
# --------------------------------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain


def save_uploaded_files(files):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    paths = []

    for file in files:
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        paths.append(path)

    return paths


def main():
    st.title("📄 FAISS + Ollama RAG (Fast)")
    st.caption("Upload PDFs → Build embeddings → Ask questions")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "chain" not in st.session_state:
        st.session_state.chain = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Upload PDFs and ask questions 😊"}
        ]

    with st.sidebar:
        st.header("📂 Upload PDFs")
        files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        process_clicked = st.button("🚀 Process Files")

    if process_clicked:
        if not files:
            st.warning("Please upload at least one PDF.")
            st.stop()

        with st.spinner("Processing documents..."):
            paths = save_uploaded_files(files)
            raw_text = load_documents(paths)
            chunks = split_text(raw_text)
            st.session_state.vector_store = build_vector_store(chunks)
            st.session_state.chain = build_chain(st.session_state.vector_store)

        st.success("✅ Documents indexed! Ask questions below.")

    for msg in st.session_state.messages[-2:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input(
        "Ask a question",
        disabled=st.session_state.vector_store is None,
    )

    if user_query:
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(user_query)
                st.markdown(answer)

        st.session_state.messages = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": answer},
        ]


if __name__ == "__main__":
    main()
