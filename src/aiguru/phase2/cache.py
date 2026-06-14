"""Crash-safe retrieval cache shared between Phase 2 and Phase 3."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from aiguru.phase2.retriever import ScoredChunk


class RetrievalCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._id_to_offset = {}
        self._question_to_offset = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.path.exists():
            return
        import re
        pattern = re.compile(br'^\{"id":\s*(\d+),\s*"question":\s*"((?:[^"\\]|\\.)*)"')
        with self.path.open("rb") as handle:
            offset = 0
            for line in handle:
                if not line.strip().endswith(b'}'):
                    offset += len(line)
                    continue
                m = pattern.match(line)
                if m:
                    try:
                        q_id = int(m.group(1))
                        q_text = json.loads(b'"' + m.group(2) + b'"')
                        self._id_to_offset[q_id] = offset
                        self._question_to_offset[q_text] = offset
                    except Exception:
                        # Fallback for parsing errors
                        try:
                            row = json.loads(line)
                            q_id = int(row["id"])
                            self._id_to_offset[q_id] = offset
                            self._question_to_offset[row["question"]] = offset
                        except Exception:
                            pass
                else:
                    # Fallback if regex pattern mismatch
                    try:
                        row = json.loads(line)
                        q_id = int(row["id"])
                        self._id_to_offset[q_id] = offset
                        self._question_to_offset[row["question"]] = offset
                    except Exception:
                        pass
                offset += len(line)

    @property
    def completed_ids(self) -> set[int]:
        return set(self._id_to_offset.keys())

    def append(self, question_id: int, question: str, chunks: Iterable[ScoredChunk]) -> bool:
        question_id = int(question_id)
        if question_id in self._id_to_offset:
            return False
        row = {
            "id": question_id,
            "question": question,
            "chunks": [
                {
                    "chunk_id": chunk.node["chunk_id"],
                    "text": chunk.node["text"],
                    "metadata": chunk.node.get("metadata") or {},
                    "score": chunk.score,
                }
                for chunk in chunks
            ],
        }
        row_str = json.dumps(row, ensure_ascii=False) + "\n"
        row_bytes = row_str.encode("utf-8")
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb+") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.seek(0, os.SEEK_END)
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        with self.path.open("ab") as handle:
            offset = handle.tell()
            handle.write(row_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        self._id_to_offset[question_id] = offset
        self._question_to_offset[question] = offset
        return True

    def _read_row_at_offset(self, offset: int) -> Dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline()
            return json.loads(line.decode("utf-8"))

    def retrieve_by_id(self, question_id: int, question: str | None = None) -> List[Dict[str, Any]]:
        question_id = int(question_id)
        offset = self._id_to_offset[question_id]
        row = self._read_row_at_offset(offset)
        if question is not None and row["question"] != question:
            raise ValueError(f"Retrieval cache question mismatch for id {question_id}")
        return list(row["chunks"])

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        offset = self._question_to_offset[question]
        row = self._read_row_at_offset(offset)
        return list(row["chunks"])
