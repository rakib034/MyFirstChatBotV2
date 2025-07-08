import streamlit as st
import os
from dotenv import load_dotenv
from utils import load_pdfs_from_folder, chunk_text, create_faiss_index, load_faiss_index, get_embedder
from groq import Groq

# Load API key from .env file
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
MODEL_NAME = "llama3-8b"  # use any supported model: gemma-2b-it, llama3-8b, etc.

# Check if the key is loaded
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file. Please add it.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Streamlit UI
st.title("💬 PDF Chatbot (Groq + FAISS)")
st.write("Ask questions based on your PDF knowledge base.")

# Load PDF knowledge base
if "vectorstore" not in st.session_state:
    with st.spinner("Loading documents and building vector store..."):
        text = load_pdfs_from_folder("pdfs")
        documents = chunk_text(text)
        embedder = get_embedder()
        db = create_faiss_index(documents, embedder)  # or load_faiss_index(embedder)
        st.session_state.vectorstore = db
        st.session_state.embedder = embedder

# Ask a question
query = st.text_input("Ask a question:")
if query:
    db = st.session_state.vectorstore
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the following question using the context below. 
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
"""

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        st.write(response.choices[0].message.content)
