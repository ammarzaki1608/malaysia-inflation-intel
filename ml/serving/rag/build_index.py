"""
Builds the FAISS vector index from the knowledge base.

Run this ONCE (or whenever knowledge_base.py changes) — not on every
API startup. The resulting index is saved to GCS and loaded by the
API at runtime, the same pattern as the trained XGBoost model.

Usage:
    python build_index.py
"""

import os
import json
import numpy as np
import faiss
from google import genai
from google.cloud import storage

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "modular-rush-495414-i2")
GCS_BUCKET = f"{PROJECT_ID}-ml-artifacts"
EMBEDDING_MODEL = "gemini-embedding-001"

from knowledge_base import KNOWLEDGE_BASE


def embed_texts(client: genai.Client, texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using gemini-embedding-001.

    task_type="RETRIEVAL_DOCUMENT" is used here because we're embedding
    the corpus that will be SEARCHED — this is distinct from
    RETRIEVAL_QUERY, which is used when embedding the user's live
    question at query time. Using the correct task_type for each side
    is what makes retrieval quality good; mixing them up silently
    degrades search relevance without throwing any error.
    """
    embeddings = []
    for text in texts:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        embeddings.append(result.embeddings[0].values)
    return np.array(embeddings, dtype="float32")


def main():
    client = genai.Client()

    texts = [entry["text"] for entry in KNOWLEDGE_BASE]
    ids = [entry["id"] for entry in KNOWLEDGE_BASE]

    print(f"Embedding {len(texts)} knowledge base entries...")
    embeddings = embed_texts(client, texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build a flat L2 FAISS index — appropriate for this corpus size
    # (a few dozen entries). At this scale, exact search is fast enough
    # that an approximate index would add complexity for no real benefit.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index and metadata locally first
    faiss.write_index(index, "knowledge_index.faiss")
    with open("knowledge_metadata.json", "w") as f:
        json.dump({"ids": ids, "texts": texts}, f)

    print("Uploading to GCS...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET)

    bucket.blob("rag/knowledge_index.faiss").upload_from_filename("knowledge_index.faiss")
    bucket.blob("rag/knowledge_metadata.json").upload_from_filename("knowledge_metadata.json")

    print(f"Done. Index uploaded to gs://{GCS_BUCKET}/rag/")


if __name__ == "__main__":
    main()