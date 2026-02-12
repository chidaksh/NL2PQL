from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_KB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pql_knowledge_base.json")
_REF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pql_reference.txt")

_kb_cache: list[dict[str, Any]] | None = None
_embeddings_cache: np.ndarray | None = None
_model_cache = None
_ref_cache: str | None = None


def _get_embedding_model():
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        try:
            _model_cache = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache


def _load_kb() -> list[dict[str, Any]]:
    global _kb_cache
    if _kb_cache is None:
        with open(_KB_PATH) as f:
            _kb_cache = json.load(f)
    return _kb_cache


def _build_embedding_text(entry: dict[str, Any]) -> str:
    parts = [
        entry.get("natural_language", ""),
        f"intent: {entry.get('intent', '')}",
        f"task: {entry.get('task_type', '')}",
        f"agg: {entry.get('aggregation', '')}",
        f"entity: {entry.get('entity_table', '')}.{entry.get('entity_column', '')}",
        f"target: {entry.get('target_table', '')}.{entry.get('target_column', '')}",
    ]
    return " | ".join(parts)


def _get_embeddings() -> np.ndarray:
    global _embeddings_cache
    if _embeddings_cache is None:
        kb = _load_kb()
        model = _get_embedding_model()
        texts = [_build_embedding_text(e) for e in kb]
        _embeddings_cache = model.encode(texts, normalize_embeddings=True)
    return _embeddings_cache


def load_pql_reference() -> str:
    global _ref_cache
    if _ref_cache is None:
        with open(_REF_PATH) as f:
            _ref_cache = f.read()
    return _ref_cache


def retrieve_similar_pql(
    query: str,
    schema_summary: str = "",
    top_k: int = 6,
    rfm_only: bool = True,
) -> list[dict[str, Any]]:
    kb = _load_kb()
    embeddings = _get_embeddings()
    model = _get_embedding_model()

    search_text = f"{query} | {schema_summary}" if schema_summary else query
    query_emb = model.encode([search_text], normalize_embeddings=True)

    similarities = (query_emb @ embeddings.T).flatten()

    if rfm_only:
        for i, entry in enumerate(kb):
            if not entry.get("rfm_compatible", True):
                similarities[i] *= 0.5

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        entry = dict(kb[idx])
        entry["similarity_score"] = float(similarities[idx])
        results.append(entry)

    if results:
        logger.info(f"Retrieved {len(results)} PQL examples (top score: {results[0]['similarity_score']:.3f})")
    return results


def format_examples_for_prompt(examples: list[dict[str, Any]]) -> str:
    lines = []
    for i, ex in enumerate(examples, 1):
        compat = "RFM" if ex.get("rfm_compatible") else "Enterprise-only"
        lines.append(f"Example {i} [{compat}] — {ex.get('natural_language', '')}")
        lines.append(f"  PQL: {ex['pql']}")
        lines.append(f"  Task: {ex.get('task_type', '')} | Intent: {ex.get('intent', '')}")
        if ex.get("uses_where"):
            lines.append(f"  WHERE: {ex.get('where_clause', '')}")
        if not ex.get("rfm_compatible") and ex.get("rfm_equivalent"):
            lines.append(f"  RFM equivalent: {ex['rfm_equivalent']}")
        lines.append("")
    return "\n".join(lines)
