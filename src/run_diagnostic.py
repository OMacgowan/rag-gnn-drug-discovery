import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from generation.generate import generate_no_context_answer


def run_diagnostic():
    with open(config.GOLD_QUESTIONS_PATH.replace("gold_questions.json", "diagnostic_questions.json"), "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        print(f"Asking {q['id']}: {q['question']}")
        answer = generate_no_context_answer(q["question"])
        results.append({
            "id": q["id"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "source_doc_id": q["source_doc_id"],
            "model_answer_no_context": answer,
        })
        print(f"  Model answered: {answer[:200]}\n")

    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.EVAL_RESULTS_DIR, "diagnostic_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {out_path}")
    print("Manually compare model_answer_no_context to reference_answer for each question,")
    print("and record correct/incorrect in the report to support the corpus-suitability diagnostic.")


if __name__ == "__main__":
    run_diagnostic()
