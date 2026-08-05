import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from generation.generate import generate_answer
from eval.metrics import reciprocal_rank, recall_at_k, hit_at_k, aggregate


def load_gold_questions():
    with open(config.GOLD_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_retriever_on_questions(retriever, retriever_name, questions):
    per_question = []
    for q in questions:
        retrieved = retriever.search(q["question"], top_k=config.TOP_K)
        retrieved_ids = [r["chunk"]["chunk_id"] for r in retrieved]
        gold_ids = q["source_chunk_ids"]

        rr = reciprocal_rank(retrieved_ids, gold_ids) if q["type"] != "unanswerable" else None
        rec = recall_at_k(retrieved_ids, gold_ids) if q["type"] != "unanswerable" else None
        hit = hit_at_k(retrieved_ids, gold_ids) if q["type"] != "unanswerable" else None

        answer = generate_answer(q["question"], retrieved)

        per_question.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "gold_chunk_ids": gold_ids,
            "retriever": retriever_name,
            "retrieved_chunk_ids": retrieved_ids,
            "reciprocal_rank": rr,
            f"recall_at_{config.TOP_K}": rec,
            f"hit_at_{config.TOP_K}": hit,
            "generated_answer": answer,
        })
        print(f"  [{retriever_name}] {q['id']} ({q['type']}) done")

    return per_question


def summarize(per_question, retriever_name):
    scored = [r for r in per_question if r["type"] != "unanswerable"]
    mrr = aggregate([r["reciprocal_rank"] for r in scored])
    recall = aggregate([r[f"recall_at_{config.TOP_K}"] for r in scored])
    hit = aggregate([r[f"hit_at_{config.TOP_K}"] for r in scored])
    return {
        "retriever": retriever_name,
        "num_scored_questions": len(scored),
        "mrr": mrr,
        f"recall_at_{config.TOP_K}": recall,
        f"hit_at_{config.TOP_K}": hit,
    }


def write_human_grading_csv(all_results, path):
    fieldnames = [
        "id", "type", "retriever", "question", "reference_answer",
        "generated_answer", "retrieved_chunk_ids",
        "correct_yn", "supported_by_citation_yn", "appropriate_idk_yn", "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "id": r["id"],
                "type": r["type"],
                "retriever": r["retriever"],
                "question": r["question"],
                "reference_answer": r["reference_answer"],
                "generated_answer": r["generated_answer"],
                "retrieved_chunk_ids": "; ".join(r["retrieved_chunk_ids"]),
                "correct_yn": "",
                "supported_by_citation_yn": "",
                "appropriate_idk_yn": "",
                "notes": "",
            })


def run_eval():
    questions = load_gold_questions()
    print(f"Loaded {len(questions)} gold questions")

    bm25 = BM25Retriever()
    dense = DenseRetriever()

    print("\nRunning BM25 retrieval + generation...")
    bm25_results = run_retriever_on_questions(bm25, "bm25", questions)

    print("\nRunning dense retrieval + generation...")
    dense_results = run_retriever_on_questions(dense, "dense", questions)

    all_results = bm25_results + dense_results

    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(config.EVAL_RESULTS_DIR, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results to {results_path}")

    csv_path = os.path.join(config.EVAL_RESULTS_DIR, "human_grading.csv")
    write_human_grading_csv(all_results, csv_path)
    print(f"Saved human grading sheet to {csv_path}")
    print("Open that CSV and fill in correct_yn / supported_by_citation_yn / appropriate_idk_yn (y/n) for each row.")

    bm25_summary = summarize(bm25_results, "bm25")
    dense_summary = summarize(dense_results, "dense")

    summary_path = os.path.join(config.EVAL_RESULTS_DIR, "retrieval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([bm25_summary, dense_summary], f, indent=2, ensure_ascii=False)

    print("\nRetrieval summary (factoid + multi-hop questions only):")
    for s in (bm25_summary, dense_summary):
        print(f"  {s['retriever']}: MRR={s['mrr']:.3f}  Recall@{config.TOP_K}={s[f'recall_at_{config.TOP_K}']:.3f}  Hit@{config.TOP_K}={s[f'hit_at_{config.TOP_K}']:.3f}")


if __name__ == "__main__":
    run_eval()
