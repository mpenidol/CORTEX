"""
Session logger — records token counts, timings, parameters and outputs.
Each session writes a row to results/sessions.csv and a JSON to results/<session_id>.json.
"""

import csv
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_PATH = os.path.join(RESULTS_DIR, "sessions.csv")

CSV_FIELDS = [
    "session_id",
    "timestamp",
    "arch",
    "sweep_type",
    "repetition",
    "prompt",
    "model",
    "db_size",
    "retrieval_k",
    "retrieval_top_n",
    # Semantic Agent
    "semantic_input_tokens",
    "semantic_output_tokens",
    "semantic_time_s",
    # Retrieval
    "retrieval_time_s",
    # Asset Agent
    "asset_input_tokens",
    "asset_output_tokens",
    "asset_time_s",
    # Layout
    "layout_time_s",
    # Totals
    "total_time_s",
    "objects_spawned",
    "output_json_file",
]


@dataclass
class SessionLog:
    prompt: str
    model: str
    db_size: int
    retrieval_k: int
    retrieval_top_n: int
    sweep_type: str = "manual"
    repetition: int = 1
    arch: str = "rag_rerank"

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    semantic_input_tokens: int = 0
    semantic_output_tokens: int = 0
    semantic_time_s: float = 0.0

    retrieval_time_s: float = 0.0

    asset_input_tokens: int = 0
    asset_output_tokens: int = 0
    asset_time_s: float = 0.0

    layout_time_s: float = 0.0
    total_time_s: float = 0.0
    objects_spawned: int = 0
    output_json_file: str = ""

    _start: float = field(default_factory=time.time, repr=False)

    def start_total(self):
        self._start = time.time()

    def end_total(self):
        self.total_time_s = round(time.time() - self._start, 3)


def extract_tokens(raw_response) -> tuple[int, int]:
    """Extract input/output tokens from a LangChain raw AIMessage (Ollama format)."""
    meta = getattr(raw_response, "response_metadata", {}) or {}
    input_t  = meta.get("prompt_eval_count") or 0
    output_t = meta.get("eval_count")        or 0
    return int(input_t), int(output_t)


def _csv_path_for_arch(arch: str) -> str:
    """Cada arquitectura escribe en su propio CSV para evitar conflictos de esquema."""
    if arch == "rag_rerank":
        return CSV_PATH   # compatibilidad con el CSV original
    return os.path.join(RESULTS_DIR, f"sessions_{arch}.csv")


def save_session(log: SessionLog, output_payload: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_filename = f"{log.session_id}.json"
    json_path = os.path.join(RESULTS_DIR, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    log.output_json_file = json_filename

    csv_path     = _csv_path_for_arch(log.arch)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: getattr(log, k) for k in CSV_FIELDS})

    print(f"[Logger] Session {log.session_id} saved → results/{json_filename}")
