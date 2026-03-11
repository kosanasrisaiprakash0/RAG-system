#ct - CHUNKING TEXT
def chunk_text(text, chunk_size=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

doc = "RAG helps LLMs answer using external data. It improves factual accuracy and reduces hallucinations."
chunks = chunk_text(doc, chunk_size=5)

for c in chunks:
    print(c)
