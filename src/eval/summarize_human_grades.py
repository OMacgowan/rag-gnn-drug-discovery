import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rate(rows, field):
    vals = [r[field].strip().lower() for r in rows if r[field].strip().lower() in ("y", "n")]
    if not vals:
        return None
    return sum(1 for v in vals if v == "y") / len(vals)


def summarize():
    csv_path = os.path.join(config.EVAL_RESULTS_DIR, "human_grading.csv")
    rows = load_rows(csv_path)

    by_retriever = defaultdict(list)
    for r in rows:
        by_retriever[r["retriever"]].append(r)

    summary = []
    for retriever, rrows in by_retriever.items():
        unanswerable_rows = [r for r in rrows if r["type"] == "unanswerable"]
        summary.append({
            "retriever": retriever,
            "num_questions": len(rrows),
            "answer_correctness": rate(rrows, "correct_yn"),
            "citation_faithfulness": rate(rrows, "supported_by_citation_yn"),
            "appropriate_idk_rate": rate(unanswerable_rows, "appropriate_idk_yn"),
        })

    out_path = os.path.join(config.EVAL_RESULTS_DIR, "generation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Generation evaluation summary (from human grading):\n")
    for s in summary:
        print(f"{s['retriever']}:")
        print(f"  Answer correctness:      {s['answer_correctness']:.0%}")
        print(f"  Citation faithfulness:   {s['citation_faithfulness']:.0%}")
        print(f"  Appropriate 'I don't know' rate: {s['appropriate_idk_rate']:.0%}")
        print()

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    summarize()
