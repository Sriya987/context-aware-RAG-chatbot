import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

load_dotenv()
st.set_page_config(page_title="Cloud RAG (Qdrant)", layout="wide")

# ---------------- PROMPT ----------------
prompt = PromptTemplate(
    template="""You are an expert assistant.
Answer the question in a clear, detailed manner using only the given context.
Explain concepts step by step when possible.
If the context is insufficient, say "I don't have enough information".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
)


# ---------------- LLM ----------------
@st.cache_resource
def get_llm():
    return OllamaLLM(
        model="llama3.2",
        temperature=0.3,
        base_url="http://127.0.0.1:11434",
        num_predict=256,
    )

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434"
    )

# ---------------- DOCUMENT LOAD ----------------
def load_documents(files):
    text = ""
    for file in files:
        reader = PdfReader(file)
        for p in reader.pages:
            text += p.extract_text() or ""
    return text

# ---------------- SPLIT ----------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=30,
    )
    chunks = splitter.split_text(text)
    st.info(f"📊 Created {len(chunks)} chunks")
    return chunks

# ---------------- QDRANT CLOUD VECTOR STORE ----------------
@st.cache_resource(show_spinner=False)
def build_vector_store(chunks):
    embeddings = get_embeddings()

    return QdrantVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="rag_docs",
    )

# ---------------- RAG CHAIN ----------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

# ---------------- UI ----------------
def main():
    st.title("☁️ Cloud RAG (Qdrant + Ollama)")
    st.caption("Free Cloud Vector DB • Production-style RAG")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload PDFs and ask questions 😊"}
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

        with st.spinner("Embedding & uploading to cloud DB..."):
            raw_text = load_documents(files)
            chunks = split_text(raw_text)
            st.session_state.vector_store = build_vector_store(chunks)
            st.session_state.chain = build_chain(st.session_state.vector_store)

        st.success("✅ Stored in Qdrant Cloud!")

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
