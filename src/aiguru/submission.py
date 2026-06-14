"""Strict competition submission validation and flat ZIP packaging."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

from aiguru.phase1.metadata_schema import is_submission_doc_id
from aiguru.phase4.postprocess import extract_article_numbers

REQUIRED_FIELDS = {"id", "question", "answer", "relevant_docs", "relevant_articles"}
ARTICLE_FORMAT = re.compile(r"^[^|]+\|[^|]+\|Điều \d+[A-Za-z]?$")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_results(
    results: Sequence[Dict[str, Any]],
    questions: Sequence[Dict[str, Any]],
) -> List[str]:
    errors: List[str] = []
    if not isinstance(results, list):
        return ["results.json root must be a JSON array"]
    if not isinstance(questions, list):
        return ["questions file root must be a JSON array"]
    question_map = {int(item["id"]): str(item["question"]) for item in questions}
    seen = set()
    if len(results) != len(questions):
        errors.append(f"expected {len(questions)} entries, found {len(results)}")

    for index, result in enumerate(results):
        prefix = f"entry[{index}]"
        missing = REQUIRED_FIELDS - set(result)
        if missing:
            errors.append(f"{prefix}: missing fields {sorted(missing)}")
            continue
        try:
            result_id = int(result["id"])
        except (TypeError, ValueError):
            errors.append(f"{prefix}: id must be integer")
            continue
        if result_id in seen:
            errors.append(f"{prefix}: duplicate id {result_id}")
        seen.add(result_id)
        if result_id not in question_map:
            errors.append(f"{prefix}: unknown id {result_id}")
        elif result["question"] != question_map[result_id]:
            errors.append(f"{prefix}: question does not match source for id {result_id}")
        if not isinstance(result["answer"], str) or not result["answer"].strip():
            errors.append(f"{prefix}: answer is empty")
        if not isinstance(result["relevant_docs"], list) or not isinstance(result["relevant_articles"], list):
            errors.append(f"{prefix}: relevant fields must be lists")
            continue
        for doc in result["relevant_docs"]:
            parts = str(doc).split("|")
            if len(parts) != 2 or not is_submission_doc_id(parts[0]) or not parts[1].strip():
                errors.append(f"{prefix}: invalid relevant_doc {doc!r}")
        for article in result["relevant_articles"]:
            parts = str(article).split("|")
            if not ARTICLE_FORMAT.match(str(article)) or not is_submission_doc_id(parts[0]):
                errors.append(f"{prefix}: invalid relevant_article {article!r}")
        answer_articles = {a.upper() for a in extract_article_numbers(result["answer"])}
        missing_answer_articles = sorted({
            str(article).split("|")[-1]
            for article in result["relevant_articles"]
            if str(article).split("|")[-1].upper() not in answer_articles
        })
        if missing_answer_articles:
            errors.append(
                f"{prefix}: relevant articles missing from answer citations: "
                f"{missing_answer_articles}"
            )
        doc_set = set(result["relevant_docs"])
        for article in result["relevant_articles"]:
            article_doc = "|".join(str(article).split("|")[:2])
            if article_doc not in doc_set:
                errors.append(f"{prefix}: article document missing from relevant_docs: {article_doc!r}")
    missing_ids = sorted(set(question_map) - seen)
    if missing_ids:
        errors.append(f"missing ids: {missing_ids[:20]}")
    return errors


def validate_and_package(
    results_path: str | Path,
    questions_path: str | Path,
    zip_path: str | Path,
) -> Path:
    results_path = Path(results_path)
    errors = validate_results(load_json(results_path), load_json(questions_path))
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:50])
        raise ValueError(f"Submission validation failed with {len(errors)} error(s):\n{preview}")
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(results_path, arcname="results.json")
    return zip_path
