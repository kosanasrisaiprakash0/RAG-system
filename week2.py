from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import faiss
import numpy as np
import os

doc =  """At exactly 3:17 a.m., the streetlight outside Apartment 402 flickered and went out.Ravi noticed because that’s when his alarm clock stopped ticking.Silence flooded the room so suddenly it felt loud. He lay still, counting his breaths, when his phone buzzed once on the table.Unknown NumberDon’t look outside.Ravi laughed under his breath. Power cuts were normal. Prank messages weren’t new. Still, his hand hesitated before pulling the curtain aside.The street was wrong.The road was there. The buildings were there. But nothing cast a shadow, not even the moon. The streetlight, though clearly off, glowed faintly—like it was remembering how to be on.His phone buzzed again.You already looked, didn’t you?Ravi stepped back. “This isn’t real,” he said, because saying it felt safer than thinking.A knock came from the door.Three slow taps.He checked the peephole.The hallway was empty, except for a dark smear on the floor that hadn’t been there before. It moved—not forward, not backward—but closer, as if distance itself was folding.His phone vibrated violently now.It knows you can see it.The knocking stopped.The clock on the wall started ticking again.3:17 a.m. From the other side of the door, something whispered—using his own voice: “Ravi… open up. You’re late.”"""

def text_chunks(text, chunk_size=250, overlap=15):
    words = text.split()
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

def build_faiss(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index

if doc is not None:
    chunks = text_chunks(doc, 20, 2)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")
    index = build_faiss(embeddings)
    query = input("ask any question regarding the text given:")
    if query:
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding).astype("float32")

        k = min(2, len(chunks))
        distances, indices = index.search(query_embedding, k)
        retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        context = '\n'.join(retrieved_chunks)
        message = [
            {
                "role": "system",
                "content": (
                    "You must answer using ONLY the provided context. "
                    "If the answer is not present in the context or you are not confident, reply exactly: \"I don't know.\" "
                    "Do not fabricate information."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer based ONLY on context."
            }
        ]
        # call llm
        response = llm_client.chat_completion(
            messages=message,
            max_tokens=200,
            temperature=0.2
        )

        answer = response.choices[0].message["content"].strip()

        # check for uncertainty (case-insensitive)
        uncertainty_keywords = [
            "i don't know",
            "i cannot",
            "not mentioned",
            "no information",
            "no context",
            "unclear"
        ]
        answer_lower = answer.lower()
        is_uncertain = any(keyword in answer_lower for keyword in uncertainty_keywords)
        if is_uncertain or len(answer_lower) < 5:
            print("answer:\nI don't know")
        else:
            print(f"answer:\n{answer}")

else:
    print("I don't know")
