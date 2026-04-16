"""
Batch experiment runner for CORTEX pipeline.

Para cada sweep se generan TODAS las combinaciones de variables fijas:
  - db_sweep : varia db_size, fija (K, N)  para todo (K,N) con K>N
  - k_sweep  : varia K,       fija (db, N) para todo (db,N) con K>N
  - n_sweep  : varia N,       fija (db, K) para todo (db,K) con K>N

Cada configuración se repite REPETITIONS veces.

Usage:
  python3 batch_experiments.py
  python3 batch_experiments.py --provider vllm --model Qwen/Qwen2.5-72B-Instruct --port 8000
  python3 batch_experiments.py --json_read          # reanuda leyendo completed_runs.json
"""

import argparse
import itertools
import json
import os

from schemas import UnityMessage
from retrieval import init_retrieval
from database import ASSET_DATABASE
from logger import RESULTS_DIR

CORTEX_DIR = os.path.dirname(__file__)

# ── Prompt y configuración ─────────────────────────────────────────────────────
TEST_PROMPT = "Create a Study Room"
REPETITIONS = 10

# ── Valores posibles de cada variable ─────────────────────────────────────────
DB_SIZES = [100, 500, 1000, 2000, 3000, None]   # None = all
K_VALUES = [5, 10, 20, 50]
N_VALUES = [1, 3, 5, 10, 20]

# ── Ruta del JSON de progreso (fuera de results/ para que Git lo trackee) ──────
COMPLETED_JSON = os.path.join(CORTEX_DIR, "completed_runs.json")

# ─────────────────────────────────────────────────────────────────────────────


def _key(sweep, db, k, n, rep):
    return {"sweep": sweep, "db": db, "k": k, "n": n, "rep": rep}


def _key_tuple(entry: dict):
    return (entry["sweep"], entry["db"], entry["k"], entry["n"], entry["rep"])


# ── JSON tracker ──────────────────────────────────────────────────────────────

def load_completed_json() -> set:
    """Load completed run keys from completed_runs.json."""
    if not os.path.exists(COMPLETED_JSON):
        return set()
    with open(COMPLETED_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return {_key_tuple(e) for e in data}


def append_completed_json(sweep, db, k, n, rep):
    """Append one completed run to completed_runs.json (atomic load-append-save)."""
    if os.path.exists(COMPLETED_JSON):
        with open(COMPLETED_JSON, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(_key(sweep, db, k, n, rep))
    with open(COMPLETED_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── CSV fallback ──────────────────────────────────────────────────────────────

def load_completed_csv() -> set:
    """Load completed run keys from sessions.csv (default resume mode)."""
    import csv
    from logger import CSV_PATH
    completed = set()
    if not os.path.exists(CSV_PATH):
        return completed
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            db_val = row["db_size"]
            key = (
                row["sweep_type"],
                int(db_val) if db_val not in ("", "None") else None,
                int(row["retrieval_k"]),
                int(row["retrieval_top_n"]),
                int(row["repetition"]),
            )
            completed.add(key)
    return completed


# ── Experimentos ──────────────────────────────────────────────────────────────

def build_experiments():
    experiments = []

    # 1. db_sweep: varia db_size, fija todas las (K, N) con K > N
    for k, n in itertools.product(K_VALUES, N_VALUES):
        if k > n:
            for db in DB_SIZES:
                experiments.append(("db_sweep", db, k, n))

    # 2. k_sweep: varia K, fija todas las (db, N) — K>N se comprueba al añadir
    for db, n in itertools.product(DB_SIZES, N_VALUES):
        for k in K_VALUES:
            if k > n:
                experiments.append(("k_sweep", db, k, n))

    # 3. n_sweep: varia N, fija todas las (db, K) — K>N se comprueba al añadir
    for db, k in itertools.product(DB_SIZES, K_VALUES):
        for n in N_VALUES:
            if k > n:
                experiments.append(("n_sweep", db, k, n))

    return experiments


def run_single(db_size, k, n, sweep_type, repetition, model_name):
    import main as m
    m.DB_SIZE         = db_size
    m.RETRIEVAL_K     = k
    m.RETRIEVAL_TOP_N = n
    m.MODEL_NAME      = model_name
    m.CURRENT_SWEEP   = sweep_type
    m.CURRENT_REP     = repetition
    init_retrieval(db_size)
    actual_db = db_size if db_size is not None else len(ASSET_DATABASE)
    print(f"  DB={actual_db}  K={k}  N={n}  sweep={sweep_type}  rep={repetition}")
    return m.generate_scene(UnityMessage(content=TEST_PROMPT))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CORTEX batch experiment runner")
    parser.add_argument("--provider",  default="ollama",       help="LLM provider: ollama | vllm  (default: ollama)")
    parser.add_argument("--model",     default="gpt-oss:20b",  help="Model name  (default: gpt-oss:20b)")
    parser.add_argument("--port",      type=int, default=None,  help="Server port (default: 11434 for ollama, 8000 for vllm)")
    parser.add_argument("--json_read", action="store_true",     help="Resume using completed_runs.json instead of sessions.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    # Inicializar agentes con el proveedor/modelo/puerto indicados
    import agents
    agents.init_agents(provider=args.provider, model=args.model, port=args.port)

    experiments = build_experiments()
    total = len(experiments) * REPETITIONS

    db_c = len([e for e in experiments if e[0] == "db_sweep"])
    k_c  = len([e for e in experiments if e[0] == "k_sweep"])
    n_c  = len([e for e in experiments if e[0] == "n_sweep"])
    print(f"Configuraciones: db_sweep={db_c}  k_sweep={k_c}  n_sweep={n_c}")
    print(f"Total runs: {len(experiments)} configs × {REPETITIONS} reps = {total}")

    if args.json_read:
        completed = load_completed_json()
        print(f"Reanudando desde completed_runs.json ({len(completed)} runs completados)\n")
    else:
        completed = load_completed_csv()
        print(f"Reanudando desde sessions.csv ({len(completed)} runs completados)\n")

    skipped = 0
    run = 0
    for sweep, db, k, n in experiments:
        for rep in range(1, REPETITIONS + 1):
            run += 1
            key = (sweep, db, k, n, rep)
            if key in completed:
                skipped += 1
                continue
            print(f"[{run}/{total}] sweep={sweep}  DB={db}  K={k}  N={n}  rep={rep}")
            try:
                run_single(db, k, n, sweep, rep, args.model)
                append_completed_json(sweep, db, k, n, rep)
            except Exception as e:
                print(f"  ERROR: {e}")

    executed = run - skipped
    print(f"\nAll {total} runs processed ({skipped} skipped, {executed} executed). Results in CORTEX/results/")


if __name__ == "__main__":
    main()
