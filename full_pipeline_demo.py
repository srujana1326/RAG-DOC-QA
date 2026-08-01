from retrieve_with_guardrail import retrieve

def generate_answer_demo(question, retrieved_chunks):
    """
    STAND-IN for the real OpenAI call (see generate_answer_openai.py for
    the real version). This just neatly formats what was retrieved, so we
    can see the full pipeline logic run today without needing internet
    access from this sandbox. A real AI model would read these same
    chunks and write a proper sentence answering the question.
    """
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find relevant information in the documents to answer that question.",
            "sources": []
        }
    combined = " ".join(c["text"].replace("\n", " ") for c in retrieved_chunks)
    return {
        "answer": f"[DEMO MODE -- would be AI-written] Based on the retrieved content: {combined[:250]}...",
        "sources": [c["source"] for c in retrieved_chunks]
    }


if __name__ == "__main__":
    test_questions = [
        "What legal issues is Apple facing?",
        "What was Apple's research and development spending trend?",
        "What is the best recipe for chocolate chip cookies?",
    ]

    for q in test_questions:
        chunks = retrieve(q, distance_threshold=1.7)
        result = generate_answer_demo(q, chunks)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 60)
