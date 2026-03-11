import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import numpy as np
import faiss
import tempfile

# ------------------ UI CONFIG ------------------
st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("📄 RAG PDF Question Answering")

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

llm_client = InferenceClient(
    model="model_name here",
    api_key="token_here"
)

# ------------------ FUNCTIONS ------------------
def load_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=250):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index

# ------------------ SIDEBAR ------------------
st.sidebar.header("📂 Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# ------------------ MAIN LOGIC ------------------
if uploaded_pdf:
    with st.spinner("Reading PDF..."):
        pdf_text = load_pdf_text(uploaded_pdf)

    chunks = chunk_text(pdf_text)
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings)

    index = build_faiss_index(embeddings)

    st.success("PDF processed and indexed!")

    query = st.text_input("Ask a question from the PDF")

    if query:
        with st.spinner("Retrieving answer..."):
            query_embedding = embedder.encode([query])
            query_embedding = np.array(query_embedding).astype("float32")

            k = 2
            _, indices = index.search(query_embedding, k)
            retrieved_chunks = [chunks[i] for i in indices[0]]

            context = "\n".join(retrieved_chunks)

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers strictly from context."
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{query}

Answer using ONLY the context.
"""
                }
            ]

            response = llm_client.chat_completion(
                messages=messages,
                max_tokens=150,
                temperature=0.2,
                
            )

            answer = response.choices[0].message["content"]

        st.subheader("✅ Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Context"):
            for c in retrieved_chunks:
                st.markdown(f"- {c}")

else:
    st.info("Please upload a PDF to begin.")
