import os
import streamlit as st
import requests
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

# ---- Caching: LLM and Embeddings ----
@st.cache_resource
def get_llm():
    return OllamaLLM(
        model="neural-chat:latest",
        temperature=0.3,
        base_url="http://127.0.0.1:11434",
        num_predict=256,
        k=5,
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

prompt = PromptTemplate(
    template=(
        """
Answer briefly based only on the context.
If the context is insufficient, say "I don't have enough information".

Context:
{context}

Question:
{question}
"""
    ),
    input_variables=["context", "question"],
)


def load_documents(file_paths):
    texts = []
    for path in file_paths:
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            text += p.extract_text() or ""
        texts.append(text)
    return "\n\n".join(texts)


def split_text(text):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400,
        chunk_overlap=30,
    )
    chunks = splitter.split_text(text)
    return chunks


@st.cache_data(show_spinner=False)
def build_vector_store(chunks):
    embeddings = get_embeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)


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


def get_file_path(upload):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    if isinstance(upload, str):
        return upload
    path = os.path.join(temp_dir, upload.name)
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return path


def main():
    st.set_page_config(page_title="FAISS Neural-Chat RAG", layout="wide")
    st.title("FAISS + Neural-Chat (Ollama) - Fast RAG")
    st.write("Upload PDF files and ask questions based on their content.")

    if "messages" not in st.session_state:
        st.session_state.messages=[
            {
                "role":"assistant",
                "content":("Hi there! How can I help you today?")
            }
        ]
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    file_upload = st.sidebar.file_uploader(
        label = "Upload PDF files",
        type=["pdf","docx","pptx","txt"],
        accept_multiple_files=True,
        key="pdf_uploader")

    # Button to explicitly process files and build embeddings
    process_clicked = st.sidebar.button("Process Files & Create Embeddings")
    if file_upload and (process_clicked or (st.session_state.vector_store is None and len(file_upload) > 0)):
        with st.spinner("📄 Processing PDFs and creating embeddings..."):
            file_paths=[get_file_path(file) for file in file_upload]
            raw_text=load_documents(file_paths)
            text_chunks=split_text(raw_text)
            st.session_state.vector_store=build_vector_store(text_chunks)
        st.success("✅ Files processed! Embeddings created. Ask your questions now.")
    
    # Toggle to show/hide chat history to keep UI clean while generating
    show_history = st.sidebar.checkbox("Show previous messages", value=False, help="Toggle chat history visibility")
    if show_history:
        messages_to_render = st.session_state.messages
    else:
        # Only render the last exchange (assistant greeting + latest user/assistant)
        messages_to_render = st.session_state.messages[-2:] if len(st.session_state.messages) > 2 else st.session_state.messages

    for msg in messages_to_render:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"]) 
    
    user_prompt = st.chat_input("Your question", disabled=st.session_state.vector_store is None)
    if user_prompt:
        if st.session_state.vector_store is None:
            st.error("❌ Please upload PDF files first!")
            return
            
        st.session_state.messages.append(
            {
                "role":"user",
                "content":user_prompt
            }
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Generating Response..."):
                # Build the chain and invoke with the user's question
                chain = build_chain(st.session_state.vector_store)
                answer = chain.invoke(user_prompt)
            
                # Keep only the latest conversation turn to avoid dimmed history
                st.session_state.messages = [
                    {
                        "role":"user",
                        "content":user_prompt
                    },
                    {
                        "role":"assistant",
                        "content":answer
                    }
                ]
                st.markdown(answer)

if __name__ == "__main__":
    main()