import chromadb
import pickle

client = chromadb.PersistentClient(path="data/vectordb")
collection = client.get_or_create_collection(name="apple_10k")

with open("data/processed/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def retrieve(question, n_results=3, distance_threshold=1.7):
    """
    Finds the most relevant chunks for a question.
    Returns an empty list if nothing is confidently relevant --
    this empty-list case is what triggers the "I don't know" guardrail.

    NOTE: distance_threshold=1.7 is a rough starting point for this
    TF-IDF demo. The real project tunes this number using the
    evaluation quiz (step 7) against real embeddings, not guessed.
    """
    question_embedding = vectorizer.transform([question]).toarray()[0].tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=n_results)

    relevant = []
    for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        if dist <= distance_threshold:
            relevant.append({"text": doc, "distance": dist, "source": meta["source"]})
    return relevant


def answer_question(question):
    chunks = retrieve(question)
    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the documents to answer that question.",
            "sources": []
        }
    # (Next step will hand these chunks to an AI model to actually write
    # a real sentence. For now, just show that we correctly FOUND them.)
    return {
        "answer": f"[Would generate an answer using {len(chunks)} retrieved chunk(s)]",
        "sources": [c["source"] for c in chunks]
    }


if __name__ == "__main__":
    test_questions = [
        "What legal issues is Apple facing?",
        "What is the best recipe for chocolate chip cookies?",
    ]
    for q in test_questions:
        result = answer_question(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources found: {len(result['sources'])}")
        print()
