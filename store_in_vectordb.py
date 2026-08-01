import json
import numpy as np
import chromadb

# Load our chunks and their fingerprints (embeddings) from the last step
with open("data/processed/chunks.json") as f:
    chunks = json.load(f)
embeddings = np.load("data/processed/embeddings.npy")

# Create a local vector database (this is the "smart filing cabinet").
# In production, this would be Chroma Cloud or Pinecone (hosted, shared,
# accessible from a deployed website). For today's demo, a local one
# works exactly the same way conceptually, and needs no signup at all.
client = chromadb.PersistentClient(path="data/vectordb")
collection = client.get_or_create_collection(name="apple_10k")

# Store each chunk, along with its fingerprint and a bit of metadata
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=embeddings.tolist(),
    documents=chunks,
    metadatas=[{"source": "apple_10k_2024.txt", "chunk_index": i} for i in range(len(chunks))]
)

print(f"Stored {collection.count()} chunks in the vector database.")
print("Saved to: data/vectordb/ (a real, queryable database on disk)")
