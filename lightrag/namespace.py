from __future__ import annotations

from typing import Iterable


# All namespace should not be changed
class NameSpace:
    KV_STORE_FULL_DOCS = "full_docs"
    KV_STORE_TEXT_CHUNKS = "text_chunks"
    KV_STORE_LLM_RESPONSE_CACHE = "llm_response_cache"
    KV_STORE_FULL_ENTITIES = "full_entities"
    KV_STORE_FULL_RELATIONS = "full_relations"
    KV_STORE_ENTITY_CHUNKS = "entity_chunks"
    KV_STORE_RELATION_CHUNKS = "relation_chunks"
    KV_STORE_ALIAS_TO_CANONICAL = "alias_to_canonical"
    KV_STORE_CANONICAL_TO_ALIASES = "canonical_to_aliases"
    KV_STORE_ENTITY_RESOLUTION_REVIEW = "entity_resolution_review"

    VECTOR_STORE_ENTITIES = "entities"
    VECTOR_STORE_RELATIONSHIPS = "relationships"
    VECTOR_STORE_CHUNKS = "chunks"

    GRAPH_STORE_CHUNK_ENTITY_RELATION = "chunk_entity_relation"

    DOC_STATUS = "doc_status"


def normalize_namespace(namespace: str) -> str:
    """Normalize namespace variants to the canonical underscore format."""
    return (namespace or "").strip().lower().replace("-", "_").replace(" ", "_")


def is_namespace(namespace: str, base_namespace: str | Iterable[str]):
    if isinstance(base_namespace, str):
        return normalize_namespace(namespace).endswith(normalize_namespace(base_namespace))
    return any(is_namespace(namespace, ns) for ns in base_namespace)
