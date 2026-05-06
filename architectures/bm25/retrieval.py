"""Architecture: BM25 puro (baseline léxico).

Pipeline: query → BM25 top-K → asset agent (todos los K candidatos).

  K = numero de candidatos recuperados por el indice BM25 por objeto
      y pasados directamente al asset agent (sin filtrado posterior)
  N = NO APLICA. Todos los K candidatos se pasan al LLM.

Sin embeddings ni vectorstore. La busqueda es puramente lexica (TF-IDF/BM25):
encuentra assets cuyas descripciones comparten terminos con el query.
Parámetros sweep: db_size, K
"""

import re
from concurrent.futures import ThreadPoolExecutor

from database import ASSET_DATABASE

ARCH_NAME = "bm25"

_db_subset: list[dict] = []
_bm25 = None
_asset_map: dict = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def init_retrieval(db_size: int | None = None):
    global _db_subset, _bm25, _asset_map
    from rank_bm25 import BM25Okapi

    _db_subset = ASSET_DATABASE[:db_size] if db_size else ASSET_DATABASE
    corpus     = [_tokenize(f"{a['name']} {a['description']}") for a in _db_subset]
    _bm25      = BM25Okapi(corpus)
    _asset_map = {a["db_id"]: a for a in _db_subset}
    print(f"[BM25] Índice construido: {len(_db_subset)} docs")


def _search_single(query: str, k: int = 5) -> list[dict]:
    tokens  = _tokenize(query)
    scores  = _bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    seen, candidates = set(), []
    for idx in top_idx:
        a = _db_subset[idx]
        if a["db_id"] in seen:
            continue
        seen.add(a["db_id"])
        candidates.append({
            "db_id":      a["db_id"],
            "name":       a["name"],
            "description": a["description"],
            "size":       a["size"],
            "bm25_score": float(scores[idx]),
        })
    return candidates


def _retrieve_for_object(
    instance_id: str, named_asset: str, k: int = 5, top_n: int = 0
) -> tuple[str, list[dict]]:
    candidates = _search_single(named_asset, k=k)
    return instance_id, candidates  # todos los K candidatos al asset agent


def retrieve_all_objects(scene_objects, k: int = 5, top_n: int = 3) -> dict[str, list[dict]]:
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_retrieve_for_object, obj.instance_id, obj.named_asset, k, top_n): obj
            for obj in scene_objects
        }
        for future in futures:
            instance_id, candidates = future.result()
            results[instance_id] = candidates
    return results
