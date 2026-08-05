import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(script, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    result = subprocess.run([sys.executable, os.path.join(ROOT, script)], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n{script} failed (exit code {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def main():
    print("Reproducing all results and tables for the RAG project report.")
    print("Note: this assumes data/raw/papers.jsonl (the frozen corpus snapshot) already")
    print("exists in the repo. To re-collect the corpus from the arXiv API instead, run")
    print("python src/collect_corpus.py first (not run automatically here, since arXiv")
    print("results can drift over time and would break reproducibility of the reported numbers).")

    run("src/preprocess.py", "Step 1/4: Preprocessing and chunking the corpus")
    run("src/run_diagnostic.py", "Step 2/4: Corpus-suitability diagnostic (LLM, no retrieval)")
    run("src/eval/run_eval.py", "Step 3/4: Retrieval + generation evaluation on gold questions")

    graded_path = os.path.join(ROOT, "eval", "results", "human_grading.csv")
    if os.path.exists(graded_path):
        run("src/eval/summarize_human_grades.py", "Step 4/4: Summarizing human-graded generation results")
    else:
        print(f"\n{'=' * 70}\nStep 4/4: Summarizing human-graded generation results\n{'=' * 70}")
        print(f"Skipped: {graded_path} does not exist yet.")
        print("This step requires human judgment (per the assignment spec) and cannot be")
        print("fully automated. Copy eval/results/human_grading_template.csv to")
        print("eval/results/human_grading.csv, fill in the y/n columns by hand, then run:")
        print("  python src/eval/summarize_human_grades.py")

    print("\nDone. See eval/results/ for all generated tables:")
    print("  diagnostic_results.json      - corpus suitability diagnostic")
    print("  eval_results.json            - full retrieval + generation output")
    print("  retrieval_summary.json       - MRR / Recall@5 / Hit@5 per retriever")
    print("  generation_summary.json      - correctness / faithfulness / IDK rate per retriever (after human grading)")


if __name__ == "__main__":
    main()
