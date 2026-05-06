import importlib.util
import json
import os
import time

from fastapi import FastAPI

import agents
from database import ASSET_DATABASE
from layout import calculate_unity_layout
from logger import SessionLog, extract_tokens, save_session
from schemas import UnityMessage

DB_SIZE         = 100
RETRIEVAL_K     = 20
RETRIEVAL_TOP_N = 5
MODEL_NAME      = "gpt-oss:20b"
CURRENT_SWEEP   = "manual"
CURRENT_REP     = 1
ARCH            = "rag_rerank"   # arquitectura activa

app = FastAPI()

# ── Carga dinámica del módulo de retrieval ────────────────────────────────────
_arch_module      = None
_arch_module_name = None


def _load_arch_module(arch_name: str):
    global _arch_module, _arch_module_name
    if _arch_module_name == arch_name:
        return _arch_module
    path = os.path.join(os.path.dirname(__file__), "architectures", arch_name, "retrieval.py")
    spec = importlib.util.spec_from_file_location(f"arch_{arch_name}_retrieval", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _arch_module      = mod
    _arch_module_name = arch_name
    print(f"[Arch] Módulo cargado: {arch_name}")
    return mod


def init_retrieval(db_size=None):
    _load_arch_module(ARCH)
    _arch_module.init_retrieval(db_size)


_load_arch_module(ARCH)
init_retrieval(DB_SIZE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_candidates_prompt(scene_objects, candidates_per_object: dict) -> str:
    if hasattr(_arch_module, "format_asset_prompt"):
        return _arch_module.format_asset_prompt(scene_objects, candidates_per_object)
    lines = ["Match EVERY item below. One output entry per item.\n"]
    for i, obj in enumerate(scene_objects, 1):
        lines.append(f"ITEM {i}: ID={obj.instance_id} | Requested={obj.named_asset}")
        lines.append("  Candidates (pick the best db_id from these):")
        candidates = candidates_per_object.get(obj.instance_id, [])
        if candidates:
            for c in candidates:
                lines.append(f"    * [{c['db_id']}] {c['name']}: {c['description'][:80]}")
        else:
            lines.append("    * (none — use null)")
        lines.append("")
    return "\n".join(lines)


# ── Endpoint principal ────────────────────────────────────────────────────────

@app.post("/generate_scene")
def generate_scene(message: UnityMessage):
    log = SessionLog(
        prompt          = message.content,
        model           = MODEL_NAME,
        db_size         = DB_SIZE or len(ASSET_DATABASE),
        retrieval_k     = RETRIEVAL_K,
        retrieval_top_n = RETRIEVAL_TOP_N,
        sweep_type      = CURRENT_SWEEP,
        repetition      = CURRENT_REP,
        arch            = ARCH,
    )
    log.start_total()

    print(f"\n Unity Requested: {message.content}")

    # 1. Semantic Agent
    print("1. Semantic Agent is running...")
    t0 = time.time()
    semantic_raw = agents.semantic_chain.invoke({"input": message.content})
    log.semantic_time_s = round(time.time() - t0, 3)
    log.semantic_input_tokens, log.semantic_output_tokens = extract_tokens(semantic_raw["raw"])
    semantic_result = semantic_raw["parsed"]
    print(f"   → SceneGraph objects: {[o.instance_id + '=' + o.named_asset for o in semantic_result.objects]}")

    # 2. Retrieval (delegado al módulo de arquitectura activo)
    print(f"2. Retrieving candidates [{ARCH}]...")
    t0 = time.time()
    candidates_per_object = _arch_module.retrieve_all_objects(
        semantic_result.objects, k=RETRIEVAL_K, top_n=RETRIEVAL_TOP_N
    )
    log.retrieval_time_s = round(time.time() - t0, 3)
    for iid, cands in candidates_per_object.items():
        print(f"   → {iid}: {[c['db_id'] for c in cands]}")

    # 3. Asset Agent (la arquitectura puede sobreescribir el chain)
    print("3. Asset Agent is running...")
    asset_chain = (
        _arch_module.get_asset_chain()
        if hasattr(_arch_module, "get_asset_chain")
        else agents.asset_chain
    )
    asset_prompt_text = _format_candidates_prompt(semantic_result.objects, candidates_per_object)
    t0 = time.time()
    asset_raw = asset_chain.invoke({"input": asset_prompt_text})
    log.asset_time_s = round(time.time() - t0, 3)
    log.asset_input_tokens, log.asset_output_tokens = extract_tokens(asset_raw["raw"])
    asset_result = asset_raw["parsed"]
    print(f"   → AssetMappingReport mappings: {[(m.instance_id, m.matched_db_id) for m in asset_result.mappings]}")

    # 4. Layout
    print("4. Calculating Mathematical Layout...")
    t0 = time.time()
    final_result = calculate_unity_layout(
        semantic_objects = semantic_result.objects,
        mapped_assets    = asset_result.mappings,
        database         = ASSET_DATABASE,
    )
    log.layout_time_s = round(time.time() - t0, 3)

    log.objects_spawned = len(final_result.get("objects_to_spawn", []))
    log.end_total()
    save_session(log, final_result)

    print("Final result generated and sent to Unity!")
    return final_result


if __name__ == "__main__":
    print("Test Without Unity")
    test_result = generate_scene(UnityMessage(content="Create a Study Room"))
    print("Final Result")
    print(json.dumps(test_result, indent=2))
