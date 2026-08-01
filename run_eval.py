import json
from retrieve_with_guardrail import retrieve

with open("eval_qa_pairs.json") as f:
    qa_pairs = json.load(f)


def evaluate(distance_threshold):
    """
    Runs every quiz question through retrieval at a given threshold,
    and checks two things:
    1. For answerable questions: did the correct chunk actually get retrieved?
    2. For unanswerable questions: did the guardrail correctly find nothing?
    """
    results = []
    for pair in qa_pairs:
        chunks = retrieve(pair["question"], distance_threshold=distance_threshold)
        retrieved_text = " ".join(c["text"] for c in chunks)

        if pair["should_find_answer"]:
            correct = pair["expected_keyword"] in retrieved_text
        else:
            correct = len(chunks) == 0  # guardrail should have blocked this

        results.append({
            "question": pair["question"],
            "should_find_answer": pair["should_find_answer"],
            "correct": correct
        })
    return results


print(f"{'Threshold':<12}{'Overall Accuracy':<20}{'Answerable Acc.':<20}{'Guardrail Acc.':<15}")
print("-" * 65)

for threshold in [1.0, 1.3, 1.5, 1.7, 1.9]:
    results = evaluate(threshold)
    overall = sum(r["correct"] for r in results) / len(results)
    answerable = [r for r in results if r["should_find_answer"]]
    guardrail = [r for r in results if not r["should_find_answer"]]
    answerable_acc = sum(r["correct"] for r in answerable) / len(answerable)
    guardrail_acc = sum(r["correct"] for r in guardrail) / len(guardrail)
    print(f"{threshold:<12}{overall:<20.0%}{answerable_acc:<20.0%}{guardrail_acc:<15.0%}")

print()
print("Detailed results at threshold=1.7 (our current default):")
print()
for r in evaluate(1.7):
    status = "PASS" if r["correct"] else "FAIL"
    print(f"[{status}] {r['question']}")
