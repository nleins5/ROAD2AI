from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys
from pathlib import Path

# Add scripts directory to path to import standalone modules
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_phase3_4 import (
    RetrievedChunk,
    format_legal_context,
    normalize_retrieval_results,
    PostProcessConfig,
    PostProcessor,
    extract_article_numbers,
    JsonlResultStore,
)


def chunk(article: str, score: float = 0.8, doc_id: str = "04/2017/QH14") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}-{article}",
        text=f"Nội dung của {article}",
        score=score,
        metadata={
            "doc_id": doc_id,
            "doc_type": "Luật",
            "doc_title": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
            "article_number": article,
            "formatted_doc": f"{doc_id}|Luật {doc_id} Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
            "formatted_article": f"{doc_id}|Luật {doc_id} Luật Hỗ trợ doanh nghiệp nhỏ và vừa|{article}",
        },
    )


class Phase34Tests(unittest.TestCase):
    def test_context_groups_documents_and_sorts_articles(self):
        context = format_legal_context([chunk("Điều 5"), chunk("Điều 4")])
        self.assertLess(context.index("Điều 4"), context.index("Điều 5"))
        self.assertEqual(context.count("[VĂN BẢN 1]"), 1)

    def test_extracts_and_deduplicates_article_numbers(self):
        self.assertEqual(extract_article_numbers("Điều 4, điều 5 và Điểu 4"), ["Điều 4", "Điều 5"])

    def test_postprocessor_adds_fallback_and_builds_submission_fields(self):
        processor = PostProcessor(PostProcessConfig(fallback_threshold=0.7))
        result = processor.build_result(1, "Câu hỏi", "Câu trả lời chưa dẫn nguồn.", [chunk("Điều 4")])
        self.assertIn("Cơ sở pháp lý tham chiếu: Điều 4", result["answer"])
        self.assertEqual(len(result["relevant_docs"]), 1)
        self.assertEqual(len(result["relevant_articles"]), 1)
        self.assertEqual(set(result), {"id", "question", "answer", "relevant_docs", "relevant_articles"})

    def test_postprocessor_synchronizes_all_selected_articles_into_answer(self):
        result = PostProcessor().build_result(
            1,
            "Câu hỏi",
            "Theo Điều 4, doanh nghiệp cần thực hiện nghĩa vụ.",
            [chunk("Điều 4"), chunk("Điều 5"), chunk("Điều 6")],
        )
        self.assertEqual(
            set(extract_article_numbers(result["answer"])),
            {"Điều 4", "Điều 5", "Điều 6"},
        )

    def test_duplicate_chunks_do_not_consume_minimum_article_quota(self):
        duplicate = RetrievedChunk(
            chunk_id="duplicate",
            text="Khoản khác của Điều 4",
            score=0.95,
            metadata=chunk("Điều 4").metadata,
        )
        selected = PostProcessor().select_relevant_chunks([
            chunk("Điều 4", 0.99),
            duplicate,
            chunk("Điều 5", 0.4),
            chunk("Điều 6", 0.35),
        ])
        self.assertEqual(
            [item.metadata["article_number"] for item in selected],
            ["Điều 4", "Điều 5", "Điều 6"],
        )

    def test_jsonl_store_resumes_without_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results_partial.jsonl"
            store = JsonlResultStore(path)
            result = {"id": 1, "question": "q", "answer": "a", "relevant_docs": [], "relevant_articles": []}
            self.assertTrue(store.append(result))
            self.assertFalse(store.append(result))
            resumed = JsonlResultStore(path)
            self.assertEqual(resumed.completed_ids, {1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["id"], 1)
            results_path = resumed.export_json(Path(directory) / "results.json")
            self.assertEqual(json.loads(results_path.read_text(encoding="utf-8"))[0]["id"], 1)

    def test_jsonl_store_exports_after_partial_line_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results_partial.jsonl"
            row = {"id": 1, "question": "q", "answer": "a", "relevant_docs": [], "relevant_articles": []}
            path.write_text(
                json.dumps(row) + "\n"
                + json.dumps({**row, "answer": "latest"}) + "\n"
                + '{"id": 2, "question": "unfinished"',
                encoding="utf-8",
            )
            store = JsonlResultStore(path)
            self.assertTrue(store.append({**row, "id": 2}))
            exported = store.export_json(Path(directory) / "results.json")
            rows = json.loads(exported.read_text(encoding="utf-8"))
            self.assertEqual(rows, [{**row, "answer": "latest"}, {**row, "id": 2}])

    def test_normalizes_small_rrf_scores_for_thresholding(self):
        items = [
            {"id": "a", "text": "A", "metadata": {}},
            {"id": "b", "text": "B", "metadata": {}},
        ]

        class Scored:
            def __init__(self, node, score):
                self.node = node
                self.score = score

        chunks = normalize_retrieval_results([Scored(items[0], 0.03), Scored(items[1], 0.015)])
        self.assertEqual([item.score for item in chunks], [1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
