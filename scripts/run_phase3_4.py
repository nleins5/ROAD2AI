"""Run Phase 3-4 with Qdrant hybrid retrieval and batched Unsloth inference.

This script contains all phase 3 and phase 4 logic (prompt formatting, generation,
postprocessing, streaming, and pipeline runner) consolidated in a single standalone file.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import sys
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiguru.paths import KNOWLEDGE_DIR, OUTPUT_DIR
from aiguru.phase2.cache import RetrievalCache

# =====================================================================
# Phase 3: Configuration & Prompts
# =====================================================================

@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = os.getenv(
        "AIGURU_MODEL_NAME",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    )
    max_seq_length: int = int(os.getenv("AIGURU_MAX_SEQ_LENGTH", "8192"))
    max_new_tokens: int = int(os.getenv("AIGURU_MAX_NEW_TOKENS", "1024"))
    max_context_chars: int = int(os.getenv("AIGURU_MAX_CONTEXT_CHARS", "6000"))
    batch_size: int = int(os.getenv("AIGURU_GENERATION_BATCH_SIZE", "4"))
    temperature: float = float(os.getenv("AIGURU_TEMPERATURE", "0.0"))
    top_p: float = float(os.getenv("AIGURU_TOP_P", "0.9"))
    repetition_penalty: float = float(os.getenv("AIGURU_REPETITION_PENALTY", "1.1"))
    load_in_4bit: bool = os.getenv("AIGURU_LOAD_IN_4BIT", "1") != "0"


SYSTEM_PROMPT = """Bạn là trợ lý pháp lý AI chuyên về pháp luật Việt Nam cho doanh nghiệp SME. Hãy trả lời câu hỏi của người dùng một cách trực tiếp, chính xác dựa trên tài liệu được cung cấp. Không tóm tắt tài liệu một cách chung chung mà hãy tập trung trả lời đúng vào trọng tâm câu hỏi.

QUY TẮC BẮT BUỘC:
1. Trả lời trực tiếp vào câu hỏi của người dùng bằng tiếng Việt.
2. Chỉ sử dụng thông tin trong [CONTEXT] để làm căn cứ. Không bịa đặt điều luật, số hiệu văn bản hoặc dữ kiện.
3. Chỉ được trích dẫn các mục [TRÍCH DẪN HỢP LỆ] xuất hiện trong context. Khi trích dẫn căn cứ pháp lý, phải nêu rõ Điều X và tên văn bản tương ứng.
4. Nếu context không đủ căn cứ trả lời, nói rõ hệ thống dữ liệu chưa ghi nhận quy định cụ thể.
5. Trả lời rõ ràng, thực tiễn, phù hợp với người đọc không chuyên.
6. Kết thúc bằng cảnh báo giới hạn chuẩn.
7. BẮT BUỘC viết toàn bộ câu trả lời bằng tiếng Việt. Tuyệt đối không dùng tiếng Anh."""


ANSWER_FORMAT = """Trả lời theo cấu trúc:
1. **Căn cứ pháp lý**: Liệt kê rõ Điều X của văn bản nào.
2. **Phân tích**: Áp dụng căn cứ vào tình huống được hỏi.
3. **Tư vấn sơ bộ**: Nêu hướng xử lý thực tế cho doanh nghiệp.
4. **Cảnh báo**: Cảnh báo giới hạn: Đây là tư vấn sơ bộ từ AI, doanh nghiệp cần đối chiếu văn bản gốc hoặc tham khảo chuyên gia pháp lý trước khi áp dụng."""


# =====================================================================
# Metadata & Document Helpers
# =====================================================================

DOC_ID_PATTERN = re.compile(
    r"\b(?:\d{1,4}/\d{4}/QH\d+|\d{1,4}/(?:\d{4}/)?"
    r"(?:NĐ-CP|ND-CP|[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)+))\b",
    flags=re.IGNORECASE,
)


def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_doc_id(value: Any) -> str:
    text = normalize_whitespace(value)
    match = DOC_ID_PATTERN.search(text)
    return match.group(0).upper() if match else ""


def is_submission_doc_id(value: Any) -> bool:
    return bool(normalize_doc_id(value))


# =====================================================================
# Phase 3: Text Generation & Formatting
# =====================================================================

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_retrieval_item(cls, item: Any) -> RetrievedChunk:
        """Normalize a LlamaIndex NodeWithScore, TextNode, or plain mapping."""
        node = getattr(item, "node", item)
        if isinstance(node, Mapping):
            text = str(node.get("text", ""))
            chunk_id = str(node.get("chunk_id") or node.get("id") or "")
            metadata = node.get("metadata") or {}
            raw_score = item.get("score", 0.0) if isinstance(item, Mapping) else getattr(item, "score", 0.0)
            score = float(raw_score or 0.0)
        else:
            text = str(getattr(node, "text", "") or getattr(node, "get_content", lambda: "")())
            chunk_id = str(getattr(node, "node_id", "") or getattr(node, "id_", ""))
            metadata = getattr(node, "metadata", {}) or {}
            score = float(getattr(item, "score", 0.0) or 0.0)
        return cls(
            chunk_id=chunk_id,
            text=text,
            score=score,
            metadata=metadata,
        )


def _article_sort_key(article_number: str) -> tuple[int, str]:
    match = re.search(r"(\d+)\s*([A-Za-z]?)", article_number or "")
    return (int(match.group(1)), match.group(2).lower()) if match else (10**9, article_number)


def format_legal_context(chunks: Sequence[RetrievedChunk], max_chars: int | None = None) -> str:
    """Group retrieved chunks by document so same-numbered articles stay distinct."""
    groups: OrderedDict[str, List[RetrievedChunk]] = OrderedDict()
    for chunk in chunks:
        doc_id = str(chunk.metadata.get("doc_id") or "UNKNOWN_DOC")
        groups.setdefault(doc_id, []).append(chunk)

    blocks = ["=== CƠ SỞ DỮ LIỆU THAM CHIẾU ==="]
    for index, (doc_id, doc_chunks) in enumerate(groups.items(), 1):
        first = doc_chunks[0].metadata
        doc_type = str(first.get("doc_type") or "Văn bản")
        doc_title = str(first.get("doc_title") or doc_id)
        blocks.append(f"\n[VĂN BẢN {index}]: {doc_type} {doc_id} - {doc_title}")
        for chunk in sorted(
            doc_chunks,
            key=lambda value: _article_sort_key(str(value.metadata.get("article_number") or "")),
        ):
            article = str(chunk.metadata.get("article_number") or "Nội dung liên quan")
            citation = str(chunk.metadata.get("formatted_article") or "")
            citation_line = (
                f"\n  [TRÍCH DẪN HỢP LỆ]: {citation}"
                if citation and chunk.metadata.get("submission_eligible", True)
                else ""
            )
            blocks.append(f"- Nội dung {article}: {chunk.text.strip()}{citation_line}")
    context = "\n".join(blocks)
    return context[:max_chars].rstrip() if max_chars else context


def build_chat_messages(
    question: str,
    chunks: Sequence[RetrievedChunk],
    max_context_chars: int | None = None,
) -> List[dict]:
    context = format_legal_context(chunks, max_chars=max_context_chars)
    user_prompt = f"""[CONTEXT]
{context}
[/CONTEXT]

Dựa trên tài liệu [CONTEXT] ở trên, hãy trả lời câu hỏi sau bằng tiếng Việt:
Câu hỏi: {question.strip()}

{ANSWER_FORMAT}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


class UnslothGenerator:
    """Batched Unsloth generator; model loading is lazy and isolated from tests."""

    def __init__(self, model: Any, tokenizer: Any, config: GenerationConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
            self.tokenizer.truncation_side = "right"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(cls, config: GenerationConfig | None = None) -> UnslothGenerator:
        config = config or GenerationConfig()
        
        # Check if local Ollama API is available
        import urllib.request
        import json
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.0) as response:
                if response.status == 200:
                    print("✅ Found running local Ollama API. Using Ollama for high-speed generation.")
                    return cls(model="ollama", tokenizer=None, config=config)
        except Exception:
            pass

        try:
            # Unsloth must be imported before transformers to enable its memory patches.
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_name,
                max_seq_length=config.max_seq_length,
                dtype=None,
                load_in_4bit=config.load_in_4bit,
            )
            FastLanguageModel.for_inference(model)
            return cls(model=model, tokenizer=tokenizer, config=config)
        except (ImportError, ModuleNotFoundError, RuntimeError, Exception) as e:
            print(f"⚠️ Unsloth import/load failed ({e}). Falling back to standard transformers...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            import platform

            model_name = config.model_name
            # If the model name is the unsloth 4-bit bnb model, standard transformers cannot load it on Mac/CPU easily.
            # Map to a standard compatible model.
            if "unsloth/Qwen2.5" in model_name or "bnb-4bit" in model_name:
                if platform.system() == "Darwin":
                    # For Apple Silicon, 3B or 1.5B fits and runs well in memory
                    model_name = "Qwen/Qwen2.5-3B-Instruct"
                else:
                    model_name = "Qwen/Qwen2.5-7B-Instruct"
                print(f"👉 Mapped model to standard HF model: {model_name}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Determine device
            device = "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"

            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else (torch.float16 if device == "mps" else torch.float32)
            print(f"Loading model {model_name} on device {device} with dtype {torch_dtype}...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
            )
            if device == "mps" or device == "cuda":
                if device == "mps":
                    model = model.to("mps")
                elif device == "cuda" and not hasattr(model, "hf_device_map"):
                    model = model.to("cuda")

            return cls(model=model, tokenizer=tokenizer, config=config)

    def _device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return "cuda"

    def generate(self, questions: Sequence[str], contexts: Sequence[Sequence[RetrievedChunk]]) -> List[str]:
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have equal length")

        if self.model == "ollama":
            import json
            import urllib.request
            from concurrent.futures import ThreadPoolExecutor

            def call_ollama(args):
                question, chunks = args
                messages = build_chat_messages(question, chunks, self.config.max_context_chars)
                payload = {
                    "model": "qwen2.5:3b",
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature if self.config.temperature > 0 else 0.0,
                        "top_p": self.config.top_p if self.config.temperature > 0 else 1.0,
                        "num_predict": self.config.max_new_tokens,
                    }
                }
                req = urllib.request.Request(
                    "http://localhost:11434/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                import time
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        with urllib.request.urlopen(req, timeout=120.0) as response:
                            res_data = json.loads(response.read().decode("utf-8"))
                            answer = res_data["message"]["content"]
                            return answer.strip()
                    except Exception as e:
                        print(f"⚠️ Ollama API call failed (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2.0 * (attempt + 1))
                        else:
                            print(f"❌ Ollama API call permanently failed: {e}")
                            return "Lỗi: Không thể kết nối local Ollama API."

            max_workers = max(1, self.config.batch_size)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                answers = list(executor.map(call_ollama, zip(questions, contexts)))
            return answers

        prompts = [
            self.tokenizer.apply_chat_template(
                build_chat_messages(question, chunks, self.config.max_context_chars),
                tokenize=False,
                add_generation_prompt=True,
            )
            for question, chunks in zip(questions, contexts)
        ]
        answers: List[str] = []
        for start in range(0, len(prompts), self.config.batch_size):
            answers.extend(self._generate_with_backoff(prompts[start : start + self.config.batch_size]))
        return answers

    def _generate_with_backoff(self, prompts: Sequence[str]) -> List[str]:
        try:
            return self._generate_prompt_batch(prompts)
        except RuntimeError as exc:
            message = str(exc).lower()
            if len(prompts) <= 1 or not any(token in message for token in ("out of memory", "cuda", "cublas")):
                raise
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            midpoint = len(prompts) // 2
            return self._generate_with_backoff(prompts[:midpoint]) + self._generate_with_backoff(prompts[midpoint:])

    def _generate_prompt_batch(self, prompts: Sequence[str]) -> List[str]:
        import torch

        inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length - self.config.max_new_tokens,
        ).to(self._device())
        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "repetition_penalty": self.config.repetition_penalty,
            "use_cache": True,
        }
        if self.config.temperature <= 0:
            kwargs["do_sample"] = False
        else:
            kwargs.update(
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

        input_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        generated = outputs[:, input_width:]
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        del inputs, outputs, generated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [answer.strip() for answer in decoded]


def normalize_retrieval_results(items: Iterable[Any]) -> List[RetrievedChunk]:
    chunks = [RetrievedChunk.from_retrieval_item(item) for item in items]
    max_score = max((chunk.score for chunk in chunks), default=0.0)
    # LlamaIndex RRF scores are commonly small reciprocal-rank values. Convert
    # those to a relative 0-1 confidence scale expected by Phase 4 thresholds.
    if 0 < max_score < 0.1:
        chunks = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=chunk.score / max_score,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]
    return chunks


# =====================================================================
# Phase 4: Post-Processing
# =====================================================================

ARTICLE_PATTERN = re.compile(r"(?:Điều|Điểu|điều|điểu)\s+(\d+[A-Za-z]?)", re.IGNORECASE)
STANDARD_WARNING = (
    "Cảnh báo giới hạn: Đây là tư vấn sơ bộ từ AI, doanh nghiệp cần đối chiếu "
    "văn bản gốc hoặc tham khảo chuyên gia pháp lý trước khi áp dụng."
)


@dataclass(frozen=True)
class PostProcessConfig:
    safe_threshold: float = 0.3
    high_conf_threshold: float = 0.5
    fallback_threshold: float = 0.0
    max_articles: int = 3
    max_context_chunks: int = 10
    min_high_conf_articles: int = 2
    max_fallback_citations: int = 3


def extract_article_numbers(text: str) -> List[str]:
    seen = set()
    results = []
    for value in ARTICLE_PATTERN.findall(text or ""):
        article = f"Điều {value.upper()}"
        if article not in seen:
            seen.add(article)
            results.append(article)
    return results


def _dedupe(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


class PostProcessor:
    def __init__(self, config: PostProcessConfig | None = None):
        self.config = config or PostProcessConfig()

    def select_relevant_chunks(self, chunks: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        ranked = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
        eligible = [
            chunk for chunk in ranked
            if chunk.metadata.get("formatted_article")
            and is_submission_doc_id(chunk.metadata.get("doc_id"))
            and chunk.metadata.get("submission_eligible", True)
        ]
        unique_eligible = []
        seen_articles = set()
        for chunk in eligible:
            article = str(chunk.metadata.get("formatted_article"))
            if article not in seen_articles:
                seen_articles.add(article)
                unique_eligible.append(chunk)
        eligible = unique_eligible

        selected = [chunk for chunk in eligible if chunk.score >= self.config.high_conf_threshold]
        if len(selected) < self.config.min_high_conf_articles:
            selected = [chunk for chunk in eligible if chunk.score >= self.config.safe_threshold]
        if len(selected) < self.config.min_high_conf_articles:
            selected = eligible[: self.config.min_high_conf_articles]

        return selected[: self.config.max_articles]

    @staticmethod
    def _remove_unsupported_citations(answer: str, unsupported: Sequence[str]) -> str:
        for article in unsupported:
            number = re.escape(article.split()[-1])
            answer = re.sub(
                rf"\b(?:Điều|Điểu|điều|điểu)\s+{number}\b",
                "quy định liên quan",
                answer,
                flags=re.IGNORECASE,
            )
        return answer

    def process_answer(
        self,
        answer: str,
        relevant_chunks: Sequence[RetrievedChunk],
    ) -> tuple[str, List[str]]:
        """Validate answer citations and append a grounded fallback when needed."""
        cited = extract_article_numbers(answer)
        available = {
            str(chunk.metadata.get("article_number")): str(chunk.metadata.get("formatted_article"))
            for chunk in relevant_chunks
            if chunk.metadata.get("article_number") and chunk.metadata.get("formatted_article")
        }
        hallucinated = [article for article in cited if article not in available]
        answer = self._remove_unsupported_citations(answer, hallucinated)

        top_score = max((chunk.score for chunk in relevant_chunks), default=0.0)
        if top_score >= self.config.fallback_threshold:
            references = []
            for chunk in relevant_chunks:
                article = str(chunk.metadata.get("article_number") or "")
                formatted_doc = str(chunk.metadata.get("formatted_doc") or "")
                if article and formatted_doc:
                    doc_name = formatted_doc.split("|", 1)[-1]
                    references.append(f"{article} của {doc_name}")
            references = _dedupe(references)[: self.config.max_fallback_citations]
            if references:
                answer = answer.rstrip() + "\n\nCơ sở pháp lý tham chiếu: " + "; ".join(references) + "."
        if STANDARD_WARNING.lower() not in answer.lower():
            answer = answer.rstrip() + "\n\n" + STANDARD_WARNING
        return answer, hallucinated

    def build_result(
        self,
        question_id: int,
        question: str,
        answer: str,
        chunks: Sequence[RetrievedChunk],
    ) -> Dict[str, Any]:
        selected = self.select_relevant_chunks(chunks)
        if not answer.strip():
            answer = "Hiện tại hệ thống dữ liệu chưa ghi nhận quy định pháp lý cụ thể cho tình huống này."
        answer, _hallucinated = self.process_answer(answer, selected)
        relevant_articles = _dedupe(
            [str(chunk.metadata.get("formatted_article") or "") for chunk in selected]
        )
        relevant_docs = _dedupe(
            [str(chunk.metadata.get("formatted_doc") or "") for chunk in selected]
        )
        return {
            "id": int(question_id),
            "question": question,
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        }


# =====================================================================
# Phase 4: Streaming JSONL Store
# =====================================================================

class JsonlResultStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed_ids = self._load_completed_ids()

    def _load_completed_ids(self) -> Set[int]:
        completed: Set[int] = set()
        if not self.path.exists():
            return completed
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    completed.add(int(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return completed

    @property
    def completed_ids(self) -> Set[int]:
        return set(self._completed_ids)

    def append(self, result: Dict[str, Any]) -> bool:
        result_id = int(result["id"])
        if result_id in self._completed_ids:
            return False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb+") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.seek(0, os.SEEK_END)
                    handle.write(b"\n")
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._completed_ids.add(result_id)
        return True

    def read_all(self) -> List[Dict[str, Any]]:
        rows: Dict[int, Dict[str, Any]] = {}
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    rows[int(row["id"])] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return list(rows.values())

    def export_json(self, path: str | Path) -> Path:
        """Atomically export the accumulated JSONL rows as a JSON array."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        rows = sorted(self.read_all(), key=lambda row: int(row["id"]))
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(output_path)
        return output_path


# =====================================================================
# Phase 4: Pipeline Execution Loop
# =====================================================================

def run_generation_pipeline(
    questions: Sequence[Dict[str, Any]],
    retriever: Any,
    generator: UnslothGenerator,
    postprocessor: PostProcessor,
    result_store: JsonlResultStore,
    retrieval_top_k: int = 25,
) -> int:
    """Retrieve, batch-generate, post-process, and persist unanswered questions."""
    completed_ids = result_store.completed_ids
    pending = [question for question in questions if int(question["id"]) not in completed_ids]
    written = 0
    for start in range(0, len(pending), generator.config.batch_size):
        batch = pending[start : start + generator.config.batch_size]
        contexts: List[list] = []
        for question in batch:
            if hasattr(retriever, "retrieve_by_id"):
                items = retriever.retrieve_by_id(int(question["id"]), str(question["question"]))
            else:
                items = retriever.retrieve(str(question["question"]))
            chunks = normalize_retrieval_results(items)[:retrieval_top_k]
            selected_chunks = postprocessor.select_relevant_chunks(chunks)
            contexts.append(selected_chunks)
        answers = generator.generate([str(item["question"]) for item in batch], contexts)
        for question, answer, chunks in zip(batch, answers, contexts):
            result = postprocessor.build_result(
                question_id=int(question["id"]),
                question=str(question["question"]),
                answer=answer,
                chunks=chunks,
            )
            written += int(result_store.append(result))
        print(f"Generated {start + len(batch)}/{len(pending)} answers...")
    return written


# =====================================================================
# Main CLI Entrypoint
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Competition test JSON file")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "results_partial.jsonl")
    parser.add_argument("--results-json", type=Path, default=OUTPUT_DIR / "results.json")
    parser.add_argument("--model", default=GenerationConfig().model_name)
    parser.add_argument("--batch-size", type=int, default=GenerationConfig().batch_size)
    parser.add_argument("--retrieval-top-k", type=int, default=25)
    parser.add_argument("--safe-threshold", type=float, default=0.3)
    parser.add_argument("--high-conf-threshold", type=float, default=0.5)
    parser.add_argument("--min-articles", type=int, default=3)
    parser.add_argument("--max-articles", type=int, default=10)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--retrieval-cache",
        type=Path,
        default=KNOWLEDGE_DIR / "retrieval_results.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if args.reset:
        for path in (args.output, args.results_json):
            if path.exists():
                path.unlink()

    retriever = RetrievalCache(args.retrieval_cache)
    available_questions = [q for q in questions if int(q["id"]) in retriever.completed_ids]
    if len(available_questions) < len(questions):
        print(f"⚠️ Retrieval cache is missing {len(questions) - len(available_questions)} question(s). "
              f"Running generation for {len(available_questions)} completed questions only.")
    questions = available_questions
    if not questions:
        print("❌ No completed retrieval results found in cache. Run scripts/run_phase2_retrieve.py first.")
        return

    # Load generator model (Unsloth with local Ollama or HF transformers fallback)
    config = GenerationConfig(model_name=args.model, batch_size=args.batch_size)
    generator = UnslothGenerator.from_pretrained(config)

    store = JsonlResultStore(args.output)
    written = run_generation_pipeline(
        questions=questions,
        retriever=retriever,
        generator=generator,
        postprocessor=PostProcessor(PostProcessConfig(
            safe_threshold=args.safe_threshold,
            high_conf_threshold=args.high_conf_threshold,
            min_high_conf_articles=args.min_articles,
            max_articles=args.max_articles,
            max_fallback_citations=args.max_articles,
        )),
        result_store=store,
        retrieval_top_k=args.retrieval_top_k,
    )
    store.export_json(args.results_json)
    print(f"Phase 3-4 complete: wrote {written} new results to {args.output}")
    print(f"Exported submission data to {args.results_json}")


if __name__ == "__main__":
    main()
