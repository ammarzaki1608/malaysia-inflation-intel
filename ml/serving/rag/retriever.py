"""
Retriever — loads the FAISS index from GCS and searches it against
a live user question.

This module is loaded once at API startup, the same pattern as the
XGBoost model — downloading and rebuilding the FAISS index on every
single request would be wasteful and slow.
"""

import os
import json
import tempfile
import numpy as np
import faiss
from google import genai
from google.cloud import storage

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "modular-rush-495414-i2")
GCS_BUCKET = f"{PROJECT_ID}-ml-artifacts"
EMBEDDING_MODEL = "gemini-embedding-001"

# Module-level state, loaded once
_index = None
_ids = None
_texts = None
_genai_client = None


def load_index_from_gcs():
    """
    Downloads the FAISS index and metadata from GCS into memory.
    Call this once at API startup — mirrors load_model_from_gcs()
    in api.py for the XGBoost model.
    """
    global _index, _ids, _texts, _genai_client

    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET)

    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
        bucket.blob("rag/knowledge_index.faiss").download_to_filename(f.name)
        _index = faiss.read_index(f.name)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        bucket.blob("rag/knowledge_metadata.json").download_to_filename(f.name)

    # Re-open to actually read the JSON content (download_to_filename
    # writes bytes; we need to read them back as text)
    metadata_path = f.name
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    _ids = metadata["ids"]
    _texts = metadata["texts"]

    _genai_client = genai.Client()

    print(f"[RAG] Loaded FAISS index with {_index.ntotal} entries")


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Embeds the query and returns the top_k most similar knowledge
    base entries.

    Uses task_type="RETRIEVAL_QUERY" — the counterpart to
    RETRIEVAL_DOCUMENT used when building the index. Gemini's
    embedding model produces subtly different vectors depending on
    which side of the retrieval pair you declare, which is what
    makes matching queries against documents work well.
    """
    if _index is None:
        raise RuntimeError("RAG index not loaded — call load_index_from_gcs() first")

    result = _genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config={"task_type": "RETRIEVAL_QUERY"},
    )
    query_vector = np.array([result.embeddings[0].values], dtype="float32")

    distances, indices = _index.search(query_vector, top_k)

    matches = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        matches.append({
            "id": _ids[idx],
            "text": _texts[idx],
            "distance": float(distances[0][rank]),
        })

    return matches