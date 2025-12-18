from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from pypdf import PdfReader
from dotenv import load_dotenv
import requests
import os
load_dotenv()
import logging
import streamlit as st


prompt = PromptTemplate(
    template="""Use the following context to answer the question. If you cannot answer based on the context, say "I don't have enough information."
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)
@st.cache_resource
def get_llm():
    return Ollama(model="llama3.2", temperature=0.4, base_url="http://127.0.0.1:11434")

def load_documents(file_path):
    all_text=[]
    for file in file_path:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        all_text.append(text)
    print(all_text)
    return "\n\n".join(all_text)

def split_text(text):
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,  # Reduced for faster processing
        chunk_overlap=50)  # Reduced overlap
    texts=text_splitter.split_text(text)
    st.info(f"📊 Created {len(texts)} chunks")
    return texts

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_data(show_spinner=False)
def build_vector_store(chunks):
    # Using free local embeddings - no API key needed!
    embeddings = get_embeddings()
    
    # Process with progress
    total = len(chunks)
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Creating embeddings for {total} chunks...")
    
    vector_store=FAISS.from_texts(chunks, embedding=embeddings)
    
    progress_bar.progress(1.0)
    progress_bar.empty()
    status_text.empty()
    
    return vector_store

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def rag_chain(vector_store, query):
    qa_chain=(
        {
            "context":vector_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return qa_chain.invoke(query)

def get_file_path(file_upload):
    temp_dir="temp"
    os.makedirs(temp_dir, exist_ok=True)
    if isinstance(file_upload, str):
        file_path=file_upload
    else:
        file_path=os.path.join(temp_dir, file_upload.name)
        with open(file_path, "wb") as f:
            f.write(file_upload.getbuffer())
        return file_path
    
def check_ollama_health(base_url: str = "http://127.0.0.1:11434") -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def main():
    st.title("RAG PDF Chatbot with FAISS and GPT-4o-mini")
    st.write("Upload PDF files and ask questions based on their content.")
    logging.info("Streamlit app started.")

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

    # Health check for Ollama
    if not check_ollama_health():
        st.warning("⚠️ Ollama server not reachable at 127.0.0.1:11434. Start it with 'ollama serve'.")

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
            logging.info("Generating answer...")
            with st.spinner("Generating Response..."):
                answer=rag_chain(st.session_state.vector_store, user_prompt)
            
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