import hashlib
import json
import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from database import ASSET_DATABASE

VECTORSTORE_BASE_DIR = os.path.join(os.path.dirname(__file__), "cortex_vectorstore")

# Embeddings por defecto (Ollama local)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)


def init_embeddings(provider: str = "ollama", port: int | None = None,
                    embed_model: str = "nomic-embed-text"):
    """Switch the global embeddings object.

    provider="ollama"  → OllamaEmbeddings via localhost:{port}  (default port 11434)
    provider="vllm"    → HuggingFaceEmbeddings running fully local (no server needed)
                         embed_model should be a HuggingFace model ID,
                         default: nomic-ai/nomic-embed-text-v1.5
    """
    global embeddings
    if provider == "vllm":
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_model = embed_model if "/" in embed_model else "nomic-ai/nomic-embed-text-v1.5"
        embeddings = HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"[Embeddings] HuggingFace local — {hf_model}")
    else:
        _port = port or 11434
        embeddings = OllamaEmbeddings(
            model=embed_model,
            base_url=f"http://localhost:{_port}",
        )
        print(f"[Embeddings] Ollama — {embed_model}  port={_port}")


def _store_dir(db_size: int) -> str:
    return f"{VECTORSTORE_BASE_DIR}_{db_size}"


def _hash_file(db_size: int) -> str:
    return os.path.join(_store_dir(db_size), "db_hash.txt")


def _compute_hash(subset: list[dict]) -> str:
    return hashlib.md5(json.dumps(subset, sort_keys=True).encode()).hexdigest()


def _stored_hash(db_size: int) -> str | None:
    path = _hash_file(db_size)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return None


def _save_hash(db_size: int, hash_value: str):
    with open(_hash_file(db_size), "w") as f:
        f.write(hash_value)


def _build_vectorstore(subset: list[dict], db_size: int) -> Chroma:
    store_dir = _store_dir(db_size)
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)

    documents = [
        Document(
            page_content=obj["description"],
            metadata={
                "db_id": obj["db_id"],
                "name":  obj["name"],
                "size":  str(obj["size"]),
            },
        )
        for obj in subset
    ]
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=store_dir,
    )
    _save_hash(db_size, _compute_hash(subset))
    print(f"[Vectorstore] Index built with {len(documents)} documents → {store_dir}")
    return vectorstore


def get_vectorstore(db_size: int | None = None) -> Chroma:
    """Load or build the vectorstore for a given db_size.

    db_size: number of assets to index (None = all).
    Each size has its own persistent directory so they coexist on disk.
    """
    subset = ASSET_DATABASE[:db_size] if db_size else ASSET_DATABASE
    actual_size = len(subset)
    store_dir = _store_dir(actual_size)
    current_hash = _compute_hash(subset)

    if os.path.exists(store_dir) and _stored_hash(actual_size) == current_hash:
        print(f"[Vectorstore] Loading existing index ({actual_size} docs) from {store_dir}")
        return Chroma(persist_directory=store_dir, embedding_function=embeddings)

    print(f"[Vectorstore] Building new index for {actual_size} docs...")
    return _build_vectorstore(subset, actual_size)
