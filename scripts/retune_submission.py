"""Retune article selection from retrieval cache without rerunning the LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiguru.paths import DATA_DIR, KNOWLEDGE_DIR, OUTPUT_DIR
from aiguru.phase2.cache import RetrievalCache
# Add scripts directory to path to import standalone modules
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_phase3_4 import (
    normalize_retrieval_results,
    PostProcessConfig,
    PostProcessor,
)
from aiguru.submission import validate_and_package


def write_json_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DATA_DIR / "R2AIStage1DATA.json")
    parser.add_argument("--answers", type=Path, default=OUTPUT_DIR / "results.json")
    parser.add_argument("--retrieval-cache", type=Path, default=KNOWLEDGE_DIR / "retrieval_results.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "results_retuned.json")
    parser.add_argument("--zip", type=Path, default=OUTPUT_DIR / "submission_retuned.zip")
    parser.add_argument("--retrieval-top-k", type=int, default=25)
    parser.add_argument("--safe-threshold", type=float, default=0.3)
    parser.add_argument("--high-conf-threshold", type=float, default=0.5)
    parser.add_argument("--min-articles", type=int, default=3)
    parser.add_argument("--max-articles", type=int, default=10)
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    answer_rows = json.loads(args.answers.read_text(encoding="utf-8"))
    answers = {int(row["id"]): str(row["answer"]) for row in answer_rows}
    cache = RetrievalCache(args.retrieval_cache)
    missing_answers = {int(row["id"]) for row in questions} - set(answers)
    missing_cache = {int(row["id"]) for row in questions} - cache.completed_ids
    if missing_answers or missing_cache:
        raise RuntimeError(
            f"Cannot retune: missing {len(missing_answers)} answer(s) and "
            f"{len(missing_cache)} retrieval row(s)."
        )

    processor = PostProcessor(PostProcessConfig(
        safe_threshold=args.safe_threshold,
        high_conf_threshold=args.high_conf_threshold,
        min_high_conf_articles=args.min_articles,
        max_articles=args.max_articles,
        max_fallback_citations=args.max_articles,
    ))
    results = []
    for question in questions:
        question_id = int(question["id"])
        chunks = normalize_retrieval_results(
            cache.retrieve_by_id(question_id, str(question["question"]))
        )[: args.retrieval_top_k]
        results.append(processor.build_result(
            question_id=question_id,
            question=str(question["question"]),
            answer=answers[question_id],
            chunks=chunks,
        ))

    write_json_atomic(args.output, results)
    validate_and_package(args.output, args.questions, args.zip)
    print(f"Retuned {len(results)} rows and packaged {args.zip}")


if __name__ == "__main__":
    main()
