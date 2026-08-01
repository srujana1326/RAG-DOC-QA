import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

with open("data/processed/chunks.json", "r") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# TF-IDF: a free, offline way to turn text into numbers, scoring each word
# by how distinctive it is to that chunk vs. the rest of the document set.
# (Stand-in for today's live demo. Real project code -- shown separately --
# uses OpenAI embeddings, which understand MEANING, not just word overlap.)
vectorizer = TfidfVectorizer(stop_words="english")
embeddings = vectorizer.fit_transform(chunks).toarray()

print(f"Each chunk is now represented by {embeddings.shape[1]} numbers")
print(f"Total embeddings: {embeddings.shape[0]}")

np.save("data/processed/embeddings.npy", embeddings)
import pickle
with open("data/processed/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saved embeddings + vectorizer to data/processed/")
