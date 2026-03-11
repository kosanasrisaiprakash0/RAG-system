#ec - EMBEDDINGS CREATION
from pypdf import PdfReader

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

pdf_text = load_pdf_text("/home/saiprakash/Documents/rag/sample.pdf")   # <-- your PDF
print(pdf_text[:10])

def chunk_text(text, chunk_size) :
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

doc = "RAG helps LLMs answer using external data. It improves factual accuracy and reduces hallucinations."

chunks = chunk_text(pdf_text, chunk_size=250)

for c in chunks:
    print(c)

from sentence_transformers import SentenceTransformer
import numpy as np

# Load HF embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = embedder.encode(chunks)

# Optional: convert to numpy array (good for FAISS later)
embeddings = np.array(embeddings)

print(len(embeddings))        # number of chunks
print(len(embeddings[0]))     # vector size

import faiss
import numpy as np

# embeddings from previous step
embedding_dim = embeddings.shape[1]

index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings.astype("float32"))

print("Stored vectors:", index.ntotal)

print("\n🔁 Interactive RAG started (type 'exit' to stop)\n")


query = input("Ask a question: ").strip()

    
query_embedding = embedder.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

k = 2  # number of chunks to retrieve
distances, indices = index.search(query_embedding, k)

retrieved_chunks = [chunks[i] for i in indices[0]]

print("\nRetrieved chunks:")
for c in retrieved_chunks:
    print("-", c)
context = "\n".join(retrieved_chunks)

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="model_name here",
    api_key="token_here"
)


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions strictly using the provided context."
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

response = client.chat_completion(
    messages=messages,
    max_tokens=150,
    temperature=0.2
)

answer = response.choices[0].message["content"]

print("\nFINAL ANSWER:\n", answer)
