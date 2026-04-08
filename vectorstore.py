import hashlib
import json
import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from database import ASSET_DATABASE

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "cortex_vectorstore")
HASH_FILE = os.path.join(VECTORSTORE_DIR, "db_hash.txt")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)


def _compute_db_hash() -> str:
    serialized = json.dumps(ASSET_DATABASE, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()


def _stored_hash() -> str | None:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return f.read().strip()
    return None


def _save_hash(hash_value: str):
    with open(HASH_FILE, "w") as f:
        f.write(hash_value)


def _build_vectorstore() -> Chroma:
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)

    documents = [
        Document(
            page_content=obj["description"],
            metadata={
                "db_id": obj["db_id"],
                "name": obj["name"],
                "size": str(obj["size"]),
            },
        )
        for obj in ASSET_DATABASE
    ]
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    _save_hash(_compute_db_hash())
    print(f"[Vectorstore] Index built with {len(documents)} documents.")
    return vectorstore


def get_vectorstore() -> Chroma:
    current_hash = _compute_db_hash()

    if os.path.exists(VECTORSTORE_DIR) and _stored_hash() == current_hash:
        print("[Vectorstore] Loading existing index from disk.")
        return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

    print("[Vectorstore] Changes detected in ASSET_DATABASE. Rebuilding index...")
    return _build_vectorstore()
