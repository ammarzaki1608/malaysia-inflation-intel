"""
A tiny mirror of the knowledge base, used only by the evaluation
script to translate retrieved entry IDs back into their text —
since /ask only returns IDs for traceability, not full text.

This imports directly from the real knowledge_base.py to guarantee
it never drifts out of sync with the actual production knowledge base.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml", "serving", "rag"))
from knowledge_base import KNOWLEDGE_BASE

_id_to_text = {entry["id"]: entry["text"] for entry in KNOWLEDGE_BASE}


def get_texts_for_ids(ids: list[str]) -> list[str]:
    return [_id_to_text[i] for i in ids if i in _id_to_text]