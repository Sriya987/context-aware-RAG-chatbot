import os
import logging
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from google import genai  

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")

prompt = PromptTemplate(
    template="""Use the following context to answer the question.
If you cannot answer based on the context, say "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["context", "question"],
)

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
        for page in reader.pages:
            content += page.extract_text() or ""
        texts.append(content)
    return "\n\n".join(texts)

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)
    st.info(f"📊 Created {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks):
    embeddings = get_embeddings()
    with st.spinner("Creating FAISS index..."):
        store = FAISS.from_texts(chunks, embedding=embeddings)
    return store

def generate_llm_response(prompt_text):
    response = client.models.generate_content(
        model="gemini-2.0-flash",  # ✅ FREE TIER (newest)
        contents=prompt_text
    )
    return response.text

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)

    context = format_docs(docs)

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    return generate_llm_response(final_prompt)

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
    st.title("📄 RAG PDF Chatbot (FAISS + Gemini)")
    st.caption("Upload PDFs → Build embeddings → Ask questions")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Upload PDFs and ask questions 😊"}
        ]

    # Sidebar
    with st.sidebar:
        st.header("📂 Upload PDFs")
        files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        process_clicked = st.button("🚀 Process Files")

    # Process PDFs
    if process_clicked:
        if not files:
            st.warning("Please upload at least one PDF.")
            st.stop()

        with st.spinner("Processing documents..."):
            paths = save_uploaded_files(files)
            raw_text = load_documents(paths)
            chunks = split_text(raw_text)
            st.session_state.vector_store = build_vector_store(chunks)

        st.success("✅ Documents indexed! Ask questions below.")

    # Chat UI
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
                answer = rag_chain(
                    st.session_state.vector_store,
                    user_query
                )
                st.markdown(answer)

        st.session_state.messages = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": answer},
        ]

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
