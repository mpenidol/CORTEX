"""Architecture: RAG sin ReRank.

Pipeline: query → vectorstore top-K → asset agent (todos los K candidatos).

  K = numero de candidatos recuperados del vectorstore (ChromaDB) por objeto
      y pasados directamente al asset agent (sin filtrado posterior)
  N = NO APLICA. Todos los K candidatos se pasan al LLM.

Sin cross-encoder ni filtrado K→N. El asset agent recibe los K resultados
tal cual salen del vectorstore ordenados por similitud coseno.
Parámetros sweep: db_size, K
"""

from concurrent.futures import ThreadPoolExecutor

from vectorstore import get_vectorstore

ARCH_NAME = "rag_only"

_current_db_size: int | None = None
_vectorstore = None


def init_retrieval(db_size: int | None = None):
    global _vectorstore, _current_db_size
    if _vectorstore is None or _current_db_size != db_size:
        _vectorstore = get_vectorstore(db_size)
        _current_db_size = db_size


def _search_single(query: str, k: int = 5) -> list[dict]:
    results = _vectorstore.similarity_search_with_score(query, k=k)
    seen, candidates = set(), []
    for doc, score in results:
        db_id = doc.metadata["db_id"]
        if db_id in seen:
            continue
        seen.add(db_id)
        candidates.append({
            "db_id":        db_id,
            "name":         doc.metadata["name"],
            "description":  doc.page_content,
            "size":         doc.metadata["size"],
            "vector_score": score,
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
