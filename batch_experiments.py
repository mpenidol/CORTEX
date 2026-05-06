"""
Batch experiment runner for CORTEX pipeline.

Para cada sweep se generan TODAS las combinaciones de variables fijas:
  - db_sweep : varia db_size, fija (K, N)  para todo (K,N) con K>N
  - k_sweep  : varia K,       fija (db, N) para todo (db,N) con K>N
  - n_sweep  : varia N,       fija (db, K) para todo (db,K) con K>N

Arquitecturas disponibles:
  rag_rerank  → RAG + CrossEncoder ReRank  (original)
  rag_only    → RAG sin reranking
  full_db     → DB completa en system prompt (solo db_sweep, K=N=0)
  bm25        → BM25 puro (baseline léxico)
  bm25_rag    → Híbrido BM25 + RAG con RRF

Cada configuración se repite REPETITIONS veces.

Usage:
  python3 batch_experiments.py
  python3 batch_experiments.py --arch rag_only
  python3 batch_experiments.py --arch full_db --provider vllm --model Qwen/Qwen2.5-72B-Instruct --port 8000
  python3 batch_experiments.py --json_read
"""

import argparse
import itertools
import json
import os

from schemas import UnityMessage
from database import ASSET_DATABASE

CORTEX_DIR = os.path.dirname(__file__)

# ── Prompt y configuración ─────────────────────────────────────────────────────
TEST_PROMPT = "Create a Study Room"
REPETITIONS = 10

# ── Valores posibles de cada variable ─────────────────────────────────────────
DB_SIZES = [100, 500, 1000, 2000, 3000, None]   # None = all
K_VALUES = [5, 10, 20, 50]
N_VALUES = [1, 3, 5, 10, 20]

# Arquitecturas que no usan K ni N (solo db_sweep con K=N=0)
ARCH_DB_ONLY = {"full_db"}

# Arquitecturas que usan K pero no N (sin n_sweep; N=0 en el CSV)
ARCH_K_ONLY = {"rag_only", "bm25"}

# ── Ruta del JSON de progreso ──────────────────────────────────────────────────
# Cada arquitectura tiene su propio JSON para no mezclar progreso
def _completed_json_path(arch: str) -> str:
    return os.path.join(CORTEX_DIR, f"completed_runs_{arch}.json")


# ── Clave de run ──────────────────────────────────────────────────────────────

def _key(sweep, db, k, n, rep):
    return {"sweep": sweep, "db": db, "k": k, "n": n, "rep": rep}


def _key_tuple(entry: dict):
    return (entry["sweep"], entry["db"], entry["k"], entry["n"], entry["rep"])


# ── JSON tracker ──────────────────────────────────────────────────────────────

def load_completed_json(arch: str) -> set:
    path = _completed_json_path(arch)
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {_key_tuple(e) for e in data}


def append_completed_json(arch: str, sweep, db, k, n, rep):
    path = _completed_json_path(arch)
    data = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data.append(_key(sweep, db, k, n, rep))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── CSV fallback ──────────────────────────────────────────────────────────────

def load_completed_csv(arch: str) -> set:
    import csv
    from logger import RESULTS_DIR
    csv_path = os.path.join(RESULTS_DIR, f"sessions_{arch}.csv")
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, newline="", encoding="utf-8") as f:
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

def build_experiments(arch: str) -> list[tuple]:
    """Genera la lista de configuraciones (sweep, db, k, n) para la arquitectura."""

    if arch in ARCH_DB_ONLY:
        # Solo varía db_size; K y N no aplican (se guardan como 0)
        return [("db_sweep", db, 0, 0) for db in DB_SIZES]

    if arch in ARCH_K_ONLY:
        # Varía db_size y K; N no aplica (se guarda como 0)
        experiments = []
        for k in K_VALUES:
            for db in DB_SIZES:
                experiments.append(("db_sweep", db, k, 0))
        for db in DB_SIZES:
            for k in K_VALUES:
                experiments.append(("k_sweep", db, k, 0))
        return experiments

    experiments = []

    # db_sweep: varia db_size, fija todas las (K, N) con K > N
    for k, n in itertools.product(K_VALUES, N_VALUES):
        if k > n:
            for db in DB_SIZES:
                experiments.append(("db_sweep", db, k, n))

    # k_sweep: varia K, fija todas las (db, N) con K > N
    for db, n in itertools.product(DB_SIZES, N_VALUES):
        for k in K_VALUES:
            if k > n:
                experiments.append(("k_sweep", db, k, n))

    # n_sweep: varia N, fija todas las (db, K) con K > N
    for db, k in itertools.product(DB_SIZES, K_VALUES):
        for n in N_VALUES:
            if k > n:
                experiments.append(("n_sweep", db, k, n))

    return experiments


def run_single(db_size, k, n, sweep_type, repetition, model_name, arch):
    import main as m
    m.DB_SIZE         = db_size
    m.RETRIEVAL_K     = k
    m.RETRIEVAL_TOP_N = n
    m.MODEL_NAME      = model_name
    m.CURRENT_SWEEP   = sweep_type
    m.CURRENT_REP     = repetition
    m.ARCH            = arch
    m._load_arch_module(arch)
    m.init_retrieval(db_size)
    actual_db = db_size if db_size is not None else len(ASSET_DATABASE)
    print(f"  DB={actual_db}  K={k}  N={n}  sweep={sweep_type}  rep={repetition}  arch={arch}")
    return m.generate_scene(UnityMessage(content=TEST_PROMPT))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CORTEX batch experiment runner")
    parser.add_argument("--arch",      default="rag_rerank",
                        choices=["rag_rerank", "rag_only", "full_db", "bm25", "bm25_rag"],
                        help="Arquitectura de retrieval  (default: rag_rerank)")
    parser.add_argument("--provider",  default="ollama",
                        help="LLM provider: ollama | vllm  (default: ollama)")
    parser.add_argument("--model",     default="gpt-oss:20b",
                        help="Model name  (default: gpt-oss:20b)")
    parser.add_argument("--port",      type=int, default=None,
                        help="Server port (default: 11434 ollama / 8000 vllm)")
    parser.add_argument("--repetitions", type=int, default=10,
                        help="Repeticiones por configuracion  (default: 10)")
    parser.add_argument("--json_read", action="store_true",
                        help="Reanudar leyendo completed_runs_{arch}.json en vez del CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    # Inicializar agentes y embeddings
    import agents
    import vectorstore
    agents.init_agents(provider=args.provider, model=args.model, port=args.port)
    if args.arch not in ("bm25", "full_db"):
        vectorstore.init_embeddings(provider=args.provider, port=args.port)

    experiments = build_experiments(args.arch)
    total = len(experiments) * args.repetitions

    db_c = len([e for e in experiments if e[0] == "db_sweep"])
    k_c  = len([e for e in experiments if e[0] == "k_sweep"])
    n_c  = len([e for e in experiments if e[0] == "n_sweep"])
    if args.arch in ARCH_DB_ONLY:
        print(f"Arquitectura '{args.arch}': solo db_sweep, K y N no aplican  (configs={db_c})")
    elif args.arch in ARCH_K_ONLY:
        print(f"Arquitectura '{args.arch}': db_sweep + k_sweep, N no aplica  (db={db_c}, k={k_c})")
    else:
        print(f"Configuraciones: db_sweep={db_c}  k_sweep={k_c}  n_sweep={n_c}")
    print(f"Total runs: {len(experiments)} configs x {args.repetitions} reps = {total}")

    if args.json_read:
        completed = load_completed_json(args.arch)
        print(f"Reanudando desde completed_runs_{args.arch}.json ({len(completed)} completados)\n")
    else:
        completed = load_completed_csv(args.arch)
        print(f"Reanudando desde sessions_{args.arch}.csv ({len(completed)} completados)\n")

    skipped = 0
    run     = 0
    for sweep, db, k, n in experiments:
        for rep in range(1, args.repetitions + 1):
            run += 1
            key = (sweep, db, k, n, rep)
            if key in completed:
                skipped += 1
                continue
            print(f"[{run}/{total}] sweep={sweep}  DB={db}  K={k}  N={n}  rep={rep}  arch={args.arch}")
            try:
                run_single(db, k, n, sweep, rep, args.model, args.arch)
                append_completed_json(args.arch, sweep, db, k, n, rep)
            except Exception as e:
                print(f"  ERROR: {e}")

    executed = run - skipped
    print(f"\nAll {total} runs processed ({skipped} skipped, {executed} executed). Results in CORTEX/results/")


if __name__ == "__main__":
    main()
