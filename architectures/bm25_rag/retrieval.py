"""Architecture: Híbrido BM25 + RAG con Reciprocal Rank Fusion (RRF).

Pipeline: query → BM25 top-K  ─┐
                                 ├→ RRF merge → top-N → asset agent
          query → RAG top-K    ─┘

  K = numero de candidatos recuperados por CADA retriever (BM25 obtiene K
      candidatos Y el vectorstore obtiene K candidatos de forma independiente;
      la lista combinada antes del merge puede tener hasta 2*K entradas unicas)
  N = numero de candidatos que pasan al asset agent tras la fusion RRF
      (los N con mayor score RRF de la lista combinada)

RRF: score(doc) = sum( 1 / (60 + rank + 1) ) sobre ambas listas de ranking.
Combina señal lexica (BM25) y semantica (embeddings) sin normalizar scores.
Parámetros sweep: db_size, K, N
"""

import re
from concurrent.futures import ThreadPoolExecutor

from database import ASSET_DATABASE
from vectorstore import get_vectorstore

ARCH_NAME = "bm25_rag"

_db_subset: list[dict] = []
_bm25      = None
_vectorstore = None
_current_db_size: int | None = None
_asset_map: dict = {}

RRF_K = 60  # constante estándar de RRF


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def init_retrieval(db_size: int | None = None):
    global _db_subset, _bm25, _vectorstore, _current_db_size, _asset_map
    from rank_bm25 import BM25Okapi

    _db_subset = ASSET_DATABASE[:db_size] if db_size else ASSET_DATABASE
    corpus     = [_tokenize(f"{a['name']} {a['description']}") for a in _db_subset]
    _bm25      = BM25Okapi(corpus)
    _asset_map = {a["db_id"]: a for a in _db_subset}
    print(f"[BM25+RAG] BM25 index: {len(_db_subset)} docs")

    if _vectorstore is None or _current_db_size != db_size:
        _vectorstore     = get_vectorstore(db_size)
        _current_db_size = db_size
    print(f"[BM25+RAG] Vectorstore listo")


def _bm25_search(query: str, k: int) -> list[tuple[str, int]]:
    """Retorna [(db_id, rank_0_indexed), ...] ordenados por score BM25."""
    tokens  = _tokenize(query)
    scores  = _bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    seen, results = set(), []
    for rank, idx in enumerate(top_idx):
        db_id = _db_subset[idx]["db_id"]
        if db_id not in seen:
            seen.add(db_id)
            results.append((db_id, rank))
    return results


def _rag_search(query: str, k: int) -> list[tuple[str, int]]:
    """Retorna [(db_id, rank_0_indexed), ...] ordenados por similitud coseno."""
    docs  = _vectorstore.similarity_search_with_score(query, k=k)
    seen, results = set(), []
    for rank, (doc, _score) in enumerate(docs):
        db_id = doc.metadata["db_id"]
        if db_id not in seen:
            seen.add(db_id)
            results.append((db_id, rank))
    return results


def _rrf_merge(
    bm25_ranked: list[tuple[str, int]],
    rag_ranked:  list[tuple[str, int]],
) -> list[str]:
    """Reciprocal Rank Fusion → lista de db_id ordenada por score RRF descendente."""
    scores: dict[str, float] = {}
    for db_id, rank in bm25_ranked:
        scores[db_id] = scores.get(db_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    for db_id, rank in rag_ranked:
        scores[db_id] = scores.get(db_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores, key=lambda x: -scores[x])


def _build_candidate(db_id: str) -> dict | None:
    a = _asset_map.get(db_id)
    if a is None:
        return None
    return {
        "db_id":       a["db_id"],
        "name":        a["name"],
        "description": a["description"],
        "size":        a["size"],
    }


def _retrieve_for_object(
    instance_id: str, named_asset: str, k: int = 5, top_n: int = 3
) -> tuple[str, list[dict]]:
    bm25_r = _bm25_search(named_asset, k=k)
    rag_r  = _rag_search(named_asset, k=k)
    merged = _rrf_merge(bm25_r, rag_r)
    candidates = [c for db_id in merged[:top_n] if (c := _build_candidate(db_id))]
    return instance_id, candidates


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
