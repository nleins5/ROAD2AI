import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiguru.phase2.cache import RetrievalCache
from aiguru.phase3.generator import RetrievedChunk, build_chat_messages, normalize_retrieval_results
from aiguru.phase4.postprocess import PostProcessConfig, PostProcessor

# Load dataset
with open(PROJECT_ROOT / "data/R2AIStage1DATA.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# Load cache
retriever = RetrievalCache(PROJECT_ROOT / "knowledge_store/retrieval_results.jsonl")

# Get question index 120 (the first pending one)
q = questions[120]
print(f"Question ID: {q['id']}")
print(f"Question text: {q['question']}")

items = retriever.retrieve_by_id(int(q["id"]), str(q["question"]))
chunks = normalize_retrieval_results(items)[:25]
postprocessor = PostProcessor(PostProcessConfig(max_articles=4))
selected_chunks = postprocessor.select_relevant_chunks(chunks)

messages = build_chat_messages(q["question"], selected_chunks, 3500)

payload = {
    "model": "qwen2.5:3b",
    "messages": messages,
    "stream": False,
    "options": {
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 1024,
    }
}

print("Payload size (chars):", len(json.dumps(payload)))
print("Payload keys:", list(payload.keys()))

import urllib.request
import urllib.error

req = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

import time
start_time = time.time()
try:
    with urllib.request.urlopen(req, timeout=120.0) as response:
        print("HTTP Status:", response.status)
        res_data = json.loads(response.read().decode("utf-8"))
        duration = time.time() - start_time
        print(f"Response message keys: {res_data.keys()} (took {duration:.2f} seconds)")
        print("total_duration (s):", res_data.get("total_duration", 0) / 1e9)
        print("load_duration (s):", res_data.get("load_duration", 0) / 1e9)
        print("prompt_eval_count:", res_data.get("prompt_eval_count"))
        print("prompt_eval_duration (s):", res_data.get("prompt_eval_duration", 0) / 1e9)
        print("eval_count:", res_data.get("eval_count"))
        print("eval_duration (s):", res_data.get("eval_duration", 0) / 1e9)
        print("Response content:", res_data["message"]["content"])
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print("Response body:", e.read().decode("utf-8"))
except Exception as e:
    print("General Error:", e)
