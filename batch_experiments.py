"""
Batch experiment runner for CORTEX pipeline.
Varies DB_SIZE, RETRIEVAL_K, and RETRIEVAL_TOP_N one at a time (others fixed).
Rule enforced: K > N always.
Each configuration is repeated REPETITIONS times.

Usage: python3 batch_experiments.py
"""

from schemas import UnityMessage
from retrieval import init_retrieval
from database import ASSET_DATABASE

# ── Fixed test prompt ─────────────────────────────────────────────────────────
TEST_PROMPT  = "Create a Study Room"
MODEL_NAME   = "gpt-oss:20b"
REPETITIONS  = 10

# ── Value sets ────────────────────────────────────────────────────────────────
DB_SIZES  = [100, 500, 1000, 2000, 3000, None]   # None = all
K_VALUES  = [5, 10, 20, 50]
N_VALUES  = [1, 3, 5, 10, 20]

# ── Defaults (fixed when another variable is being swept) ─────────────────────
DEFAULT_DB = 1000
DEFAULT_K  = 20
DEFAULT_N  = 5

# ─────────────────────────────────────────────────────────────────────────────

def run_single(db_size, k, n, sweep_type, repetition):
    import main as m

    m.DB_SIZE         = db_size
    m.RETRIEVAL_K     = k
    m.RETRIEVAL_TOP_N = n
    m.MODEL_NAME      = MODEL_NAME
    m.CURRENT_SWEEP   = sweep_type
    m.CURRENT_REP     = repetition

    init_retrieval(db_size)

    actual_db = db_size if db_size is not None else len(ASSET_DATABASE)
    print(f"  DB={actual_db}  K={k}  N={n}  sweep={sweep_type}  rep={repetition}")

    return m.generate_scene(UnityMessage(content=TEST_PROMPT))


def build_experiments():
    experiments = []

    # 1. Sweep DB_SIZE  (K=DEFAULT_K, N=DEFAULT_N)
    for db in DB_SIZES:
        if DEFAULT_K > DEFAULT_N:
            experiments.append(("db_sweep", db, DEFAULT_K, DEFAULT_N))

    # 2. Sweep K  (DB=DEFAULT_DB, N=DEFAULT_N)
    for k in K_VALUES:
        if k > DEFAULT_N:
            experiments.append(("k_sweep", DEFAULT_DB, k, DEFAULT_N))

    # 3. Sweep N  (DB=DEFAULT_DB, K=DEFAULT_K)
    for n in N_VALUES:
        if DEFAULT_K > n:
            experiments.append(("n_sweep", DEFAULT_DB, DEFAULT_K, n))

    return experiments


def main():
    experiments = build_experiments()
    total = len(experiments) * REPETITIONS
    print(f"Total runs: {total}  ({len(experiments)} configs × {REPETITIONS} reps)")

    run = 0
    for i, (sweep, db, k, n) in enumerate(experiments, 1):
        for rep in range(1, REPETITIONS + 1):
            run += 1
            print(f"\n[{run}/{total}] config {i}/{len(experiments)}  rep={rep}/{REPETITIONS}  sweep={sweep}")
            try:
                run_single(db, k, n, sweep, rep)
            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\nAll {total} runs done. Results saved in CORTEX/results/")


if __name__ == "__main__":
    main()
