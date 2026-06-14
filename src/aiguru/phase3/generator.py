"""Grounded prompt construction and batched Unsloth inference."""

from __future__ import annotations

import gc
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Sequence

from aiguru.phase3.config import ANSWER_FORMAT, SYSTEM_PROMPT, GenerationConfig


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_retrieval_item(cls, item: Any) -> "RetrievedChunk":
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
    groups: "OrderedDict[str, List[RetrievedChunk]]" = OrderedDict()
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
    def from_pretrained(cls, config: GenerationConfig | None = None) -> "UnslothGenerator":
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
                try:
                    with urllib.request.urlopen(req, timeout=120.0) as response:
                        res_data = json.loads(response.read().decode("utf-8"))
                        answer = res_data["message"]["content"]
                        return answer.strip()
                except Exception as e:
                    print(f"❌ Ollama API call failed: {e}")
                    return "Lỗi: Không thể kết nối local Ollama API."

            max_workers = 4
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                answers = list(executor.map(call_ollama, zip(questions, contexts)))
            return answers

        prompts = [
            self.tokenizer.apply_chat_template(
                build_chat_messages(question, chunks, self.config.max_context_chars),
                # Context is trimmed before tokenization so the system rules and
                # user question cannot silently disappear.
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
