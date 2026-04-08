from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import CrossEncoder
from vectorstore import get_vectorstore

vectorstore = get_vectorstore()

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _search_single(query: str, k: int = 5) -> list[dict]:
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [
        {
            "db_id": doc.metadata["db_id"],
            "name": doc.metadata["name"],
            "description": doc.page_content,
            "size": doc.metadata["size"],
            "vector_score": score,
        }
        for doc, score in results
    ]


def _rerank(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, f"{c['name']}: {c['description']}") for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]


def _retrieve_for_object(
    instance_id: str, named_asset: str, k: int = 5, top_n: int = 3
) -> tuple[str, list[dict]]:
    candidates = _search_single(named_asset, k=k)
    reranked = _rerank(named_asset, candidates, top_n=top_n)
    return instance_id, reranked


def retrieve_all_objects(
    scene_objects, k: int = 5, top_n: int = 3
) -> dict[str, list[dict]]:
    """For each object in the SceneGraph, retrieve and rerank top candidates.

    Returns a dict mapping instance_id → list of ranked candidates.
    """
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
