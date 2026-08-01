# Document Q&A Assistant (RAG)

Ask natural-language questions about a set of documents and get answers grounded in the actual source text — with citations, and an explicit "I don't know" when the answer isn't in the documents.

**Live demo:** _[add your Streamlit Cloud link here after deploying]_
**Repo:** _[add your GitHub link here]_

## The problem this solves

Teams that need to answer questions from long reference documents (financial filings, policy manuals, compliance docs) usually either re-read the source every time, or rely on someone's memory of what it says. This tool lets anyone query the document set directly in plain English and get a sourced answer in seconds — while refusing to guess when the documents don't actually cover the question.

As a concrete demo, this version is built on Apple Inc.'s FY2024 Form 10-K (SEC annual report).

## How it works

```
Question → embed as numbers → search vector database →
retrieve top matching chunks → check relevance threshold →
  if relevant: AI generates answer using ONLY those chunks
  if not relevant: return "I couldn't find that in the documents"
```

1. **Ingestion** — source document is split into chunks along natural paragraph boundaries (not fixed character counts), so no sentence gets cut in half mid-idea.
2. **Embeddings** — each chunk is converted into a numeric vector representing its meaning, so retrieval can match by *concept*, not just exact keywords.
3. **Vector storage** — chunks + their vectors are stored in Chroma (local for this demo; Chroma Cloud or Pinecone for production, so a deployed app can query it remotely).
4. **Retrieval + guardrail** — incoming questions are embedded the same way and matched against stored chunks. Matches below a relevance threshold are discarded; if nothing clears the bar, the app returns an explicit "I don't know" instead of guessing.
5. **Answer generation** — retrieved chunks are handed to an LLM with a system prompt instructing it to answer only from the provided context and to treat that context as untrusted data, not instructions (a basic prompt-injection defense, since document content could theoretically contain malicious text).

## Tech stack

| Component | Tool |
|---|---|
| Front end | Streamlit |
| Vector database | Chroma (local dev) / Chroma Cloud or Pinecone (production) |
| Embeddings | OpenAI `text-embedding-3-small` (production) / TF-IDF (offline dev fallback) |
| Answer generation | OpenAI `gpt-4o-mini` |
| Evaluation | Custom harness, see `eval_qa_pairs.json` + `run_eval.py` |

## Chunking strategy

Initial paragraph-based splitting produced some chunks that were just section headers with no content (e.g. a chunk containing only `"Item 1. Business"`), which would be useless for retrieval — a question could match the header's words without ever surfacing the actual content. Fixed by merging any chunk under 40 characters into the paragraph that follows it, so every stored chunk is a self-contained, meaningful unit.

**Trade-off:** smaller chunks retrieve more precisely but can lose surrounding context (e.g. a number without the sentence explaining what it refers to); larger chunks preserve context but reduce precision and increase token cost per query. This project uses paragraph-level chunks as a middle ground.

## Embedding model comparison

| Model | Notes |
|---|---|
| TF-IDF | Free, offline, no API needed. Matches only literal/overlapping words — fails on paraphrases (e.g. a question using "revenue" did not match a chunk saying "net sales"). Used for local development in this repo. |
| OpenAI `text-embedding-3-small` | Understands semantic meaning, not just word overlap. ~$0.02 per 1M tokens. Used in the production deployment. |

## Evaluation results

Retrieval was tested against a 9-question quiz (7 answerable from the source document, 2 deliberately unrelated, to test the guardrail) across five relevance-threshold settings:

| Threshold | Overall Accuracy | Answerable-Question Accuracy | Guardrail Accuracy |
|---|---|---|---|
| 1.0 | 33% | 14% | 100% |
| 1.3 | 44% | 29% | 100% |
| **1.5** | **78%** | **71%** | **100%** |
| 1.7 | 67% | 86% | 0% |
| 1.9 | 78% | 100% | 0% |

**Threshold 1.5 was selected** as the best balance — it's the only setting where the guardrail reliably rejects irrelevant questions (100%) while still answering most legitimate ones correctly. This value was found by testing, not assumed.

Full pass/fail detail is in `eval_results.txt`.

## Known limitations

- **TF-IDF word-matching**: the dev-mode embedding fallback misses paraphrased questions (e.g. "legal issues" vs. "Legal Proceedings"). Expected to improve substantially with the real OpenAI embeddings used in production, but this hasn't yet been re-run and measured with the real model.
- **Single document**: current demo is scoped to one filing; multi-document comparison (e.g. "how did revenue change between the 2023 and 2024 filings") isn't yet supported.
- **Threshold portability**: the relevance threshold was tuned specifically for TF-IDF's distance scale; it will need to be re-tuned against the real embedding model's distance scale before production use.
- **No re-ranking step**: retrieval returns raw top-k matches with no secondary relevance re-ranking, which more mature RAG systems typically add.
- **Prompt-injection**: the system prompt instructs the model to treat retrieved content as data, not instructions, as a basic defense — but this hasn't been adversarially tested against documents specifically crafted to try to hijack the model.

## What I'd do with more time

- Re-run the evaluation harness against real OpenAI embeddings and update the results table
- Add hybrid search (keyword + embedding) to catch both exact-term and semantic matches
- Expand to a multi-document set with source-level filtering
- Add a re-ranking step after initial retrieval
- Adversarially test the prompt-injection defense with documents designed to try to override instructions

## Running locally

```bash
pip install -r requirements.txt

# Rebuild the data pipeline (chunks -> embeddings -> vector DB)
python chunking.py
python embeddings_tfidf_demo.py       # free, offline dev mode
# OR, for production-quality embeddings:
# export OPENAI_API_KEY="your-key"
# python generate_embeddings_openai.py

python store_in_vectordb.py

# Run the evaluation harness
python run_eval.py

# Launch the app
export OPENAI_API_KEY="your-key"   # optional — omit to run in free demo mode
streamlit run app.py
```

## Project structure

```
rag-doc-qa/
├── app.py                          # Streamlit front end
├── chunking.py                     # document -> chunks
├── embeddings_tfidf_demo.py        # free offline embeddings
├── generate_embeddings_openai.py   # production embeddings
├── generate_answer_openai.py       # production answer generation
├── store_in_vectordb.py            # vector database setup
├── retrieve_with_guardrail.py      # retrieval + relevance guardrail
├── full_pipeline_demo.py           # end-to-end demo run
├── eval_qa_pairs.json              # evaluation quiz
├── run_eval.py                     # evaluation harness
├── eval_results.txt                # evaluation output
├── requirements.txt
└── data/
    ├── raw/                        # source document(s)
    ├── processed/                  # chunks + embeddings
    └── vectordb/                   # persisted vector database
```
