"""
Real embeddings using OpenAI's API.
Run this on YOUR OWN computer (not in this chat) -- it needs internet
access to api.openai.com, which this sandbox doesn't have.

Setup before running:
  1. pip install openai
  2. Set your key as an environment variable (never hardcode it in this file):
       Mac/Linux:  export OPENAI_API_KEY="your-key-here"
       Windows:    set OPENAI_API_KEY=your-key-here
  3. python generate_embeddings_openai.py
"""

import json
import os
import numpy as np
from openai import OpenAI

# Reads the key from your environment -- never written in this file.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

with open("data/processed/chunks.json", "r") as f:
    chunks = json.load(f)

print(f"Generating real embeddings for {len(chunks)} chunks...")

response = client.embeddings.create(
    model="text-embedding-3-small",  # cheap, strong, ~$0.02 per 1M tokens
    input=chunks
)

embeddings = np.array([item.embedding for item in response.data])

print(f"Each chunk is now represented by {embeddings.shape[1]} numbers")
print(f"(Compare: TF-IDF gave us {215} numbers based on word overlap only.")
print(f" These {embeddings.shape[1]} numbers capture actual MEANING.)")

np.save("data/processed/embeddings_openai.npy", embeddings)
print("Saved to data/processed/embeddings_openai.npy")
