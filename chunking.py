with open("data/raw/apple_10k_2024.txt", "r") as f:
    document_text = f.read()

raw_chunks = [c.strip() for c in document_text.split("\n\n") if c.strip()]

# Merge short chunks (likely headers) into the next chunk, so nothing
# stands alone without real content attached to it.
MIN_CHUNK_LENGTH = 40  # anything shorter than this is probably just a header

merged_chunks = []
buffer = ""

for chunk in raw_chunks:
    if len(buffer) > 0:
        buffer += "\n" + chunk
    else:
        buffer = chunk

    if len(buffer) >= MIN_CHUNK_LENGTH:
        merged_chunks.append(buffer)
        buffer = ""

if buffer:  # catch any leftover text at the end
    merged_chunks.append(buffer)

print(f"Before fix: {len(raw_chunks)} chunks")
print(f"After fix:  {len(merged_chunks)} chunks")
print()

for idx, chunk in enumerate(merged_chunks[:5]):
    print(f"--- Chunk {idx + 1} ({len(chunk)} characters) ---")
    print(chunk[:200] + ("..." if len(chunk) > 200 else ""))
    print()

# Save these for the next step (embeddings)
import json
with open("data/processed/chunks.json", "w") as f:
    json.dump(merged_chunks, f, indent=2)
print(f"Saved {len(merged_chunks)} chunks to data/processed/chunks.json")
