"""
Test how well the app finds the right page.

How to run:
    python evaluate.py zentora.pdf          # search tests only (free, fast)
    python evaluate.py zentora.pdf --llm    # also ask the AI (uses OpenAI, costs a little)
    python evaluate.py homecell_manual.pdf --questions eval_questions_big.json

The PDF must already be uploaded in the app.
"""

import argparse
import json

from main import retriever, vector_store, list_documents, ask_question
from config import RERANK_THRESHOLD
from llm import NOT_FOUND

K = 5              # how many results we look at
TEST_FETCH_K = 5   # small on purpose, so the 3 setups really differ


# ---------- Helpers ----------

def page_of(metadata):
    return metadata.get("page", 0) + 1


def unique(pages):
    """Remove repeated pages but keep the order."""
    seen = []
    for p in pages:
        if p not in seen:
            seen.append(p)
    return seen


def find_file_hash(file_name):
    """Find the test PDF in the database by its name."""
    for file_hash, info in list_documents().items():
        if info["name"] == file_name:
            return file_hash

    names = [info["name"] for info in list_documents().values()]
    raise SystemExit(f"'{file_name}' is not uploaded. Uploaded files: {names}")


def get_chunks(ids):
    chunk_by_id = {c["id"]: c for c in retriever.all_chunks}
    return [chunk_by_id[i] for i in ids]


def rerank(question, chunks):
    """Score chunks with the reranker, drop weak ones, return pages."""
    if not chunks:
        return []

    scores = retriever.reranker.predict([(question, c["text"]) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)

    pages = [page_of(c["metadata"]) for score, c in ranked if score >= RERANK_THRESHOLD]
    return unique(pages)[:K]


# ---------- The three setups ----------

def vector_only(question, file_hash):
    ids, _ = retriever.vector_search(question, TEST_FETCH_K, file_hash)
    pages = [page_of(c["metadata"]) for c in get_chunks(ids)]
    return unique(pages)[:K]


def vector_plus_rerank(question, file_hash):
    ids, _ = retriever.vector_search(question, TEST_FETCH_K, file_hash)
    return rerank(question, get_chunks(ids))


def hybrid_plus_rerank(question, file_hash):
    docs = retriever.retrieve(
        question, top_k=K, fetch_k=TEST_FETCH_K, file_hash=file_hash
    )
    return unique([page_of(d["metadata"]) for d in docs])


# ---------- Test 1: does search find the right page? ----------

def test_search(name, get_pages, questions, file_hash, can_reject=True):
    hit1 = hit_k = 0
    rr_total = 0
    answerable = blocked = no_answer = 0
    problems = []

    for item in questions:
        pages = get_pages(item["q"], file_hash)
        expected = item["page"]

        # Questions with no answer in the PDF
        if expected is None:
            no_answer += 1
            if not pages:
                blocked += 1
            else:
                problems.append(f"SHOULD FIND NOTHING: {item['q']} -> pages {pages}")
            continue

        # Questions with an answer in the PDF
        answerable += 1

        if expected in pages:
            rank = pages.index(expected) + 1
            hit_k += 1
            rr_total += 1 / rank
            if rank == 1:
                hit1 += 1
        else:
            problems.append(f"MISSED page {expected}: {item['q']} -> pages {pages}")

    print(f"\n=== {name} ===")
    print(f"Right page first:      {hit1}/{answerable}  ({hit1 / answerable:.0%})")
    print(f"Right page in top {K}:   {hit_k}/{answerable}  ({hit_k / answerable:.0%})")
    print(f"MRR (1.0 is best):     {rr_total / answerable:.2f}")

    if can_reject:
        print(f"No-answer blocked:     {blocked}/{no_answer}  ({blocked / no_answer:.0%})")
    else:
        print("No-answer blocked:     (this setup has no cutoff, so it never blocks)")
        problems = [p for p in problems if not p.startswith("SHOULD FIND NOTHING")]

    for p in problems:
        print("  -", p)


# ---------- Test 2: does the full app (with the AI) answer correctly? ----------

def test_full_app(questions, file_hash):
    good = 0
    problems = []

    for item in questions:
        answer = ask_question(item["q"], file_hash=file_hash)["answer"]
        said_not_found = NOT_FOUND in answer

        if item["page"] is None:
            ok = said_not_found          # should say "not found"
        else:
            ok = not said_not_found      # should give an answer

        if ok:
            good += 1
        else:
            problems.append(f"{item['q']} -> {answer[:100]}")

    print("\n=== Full app (search + AI) ===")
    print(f"Correct behaviour: {good}/{len(questions)}  ({good / len(questions):.0%})")
    print("(Answer questions should get an answer. No-answer questions should get 'not found'.)")

    for p in problems:
        print("  - WRONG:", p)


# ---------- Run ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="name of the uploaded test PDF, e.g. zentora.pdf")
    parser.add_argument("--llm", action="store_true", help="also test the full app with the AI")
    parser.add_argument("--questions", default="eval_questions.json", help="questions file to use")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)

    retriever.refresh_index()
    file_hash = find_file_hash(args.file_name)

    test_search("1. Vector only", vector_only, questions, file_hash, can_reject=False)
    test_search("2. Vector + reranker", vector_plus_rerank, questions, file_hash)
    test_search("3. Hybrid (vector + keyword) + reranker", hybrid_plus_rerank, questions, file_hash)

    if args.llm:
        test_full_app(questions, file_hash)