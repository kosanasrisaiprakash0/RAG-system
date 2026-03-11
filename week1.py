from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import numpy as np
import faiss

doc = """At exactly 3:17 a.m., the streetlight outside Apartment 402 flickered and went out.Ravi noticed because that’s when his alarm clock stopped ticking.Silence flooded the room so suddenly it felt loud. He lay still, counting his breaths, when his phone buzzed once on the table.Unknown NumberDon’t look outside.Ravi laughed under his breath. Power cuts were normal. Prank messages weren’t new. Still, his hand hesitated before pulling the curtain aside.The street was wrong.The road was there. The buildings were there. But nothing cast a shadow, not even the moon. The streetlight, though clearly off, glowed faintly—like it was remembering how to be on.His phone buzzed again.You already looked, didn’t you?Ravi stepped back. “This isn’t real,” he said, because saying it felt safer than thinking.A knock came from the door.Three slow taps.He checked the peephole.The hallway was empty, except for a dark smear on the floor that hadn’t been there before. It moved—not forward, not backward—but closer, as if distance itself was folding.His phone vibrated violently now.It knows you can see it.The knocking stopped.The clock on the wall started ticking again.3:17 a.m. From the other side of the door, something whispered—using his own voice: “Ravi… open up. You’re late.”"""

def chunks_text(doc,chunk_size = 250,overlap = 10):
    words = doc.split()
    chunks = []
    for i in range(0,len(words),chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

llm_client = InferenceClient(
    model="model_name here",
    api_key="token_here"
)

def faissing(embedding):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embedding.astype("float32"))
    return index

if doc is not None:
    chunks = chunks_text(doc,20,2)
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings)
    
    index = faissing(embeddings)
    query = input("Ask a question about the text you have given")
    if query is not None:
        query_embedding = embedder.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        k = 2
        distances,indices = index.search(query_embedding,k)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        print(f"{[k for k in retrieved_chunks]}")
    # else:
    #     print("Ask a question about the text you have given")
else:
    print("provide info to ask a query")

