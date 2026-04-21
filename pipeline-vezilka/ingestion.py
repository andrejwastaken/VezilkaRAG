"""
Phase 6 – LightRAG Ingestion
==============================

Inserts normalized (and optionally LLM-summarized) documents into LightRAG
along with structured metadata for filtered retrieval.

Requirements:
  - OPENAI_API_KEY must be set in the environment (or a compatible LLM configured)
  - lightrag package must be installed  (pip install -e .[api])
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_WORKING_DIR = Path(os.getenv("WORKING_DIR", str(DATA_DIR / "rag_storage")))
DEFAULT_SPLIT_BY_CHARACTER = "\n\n"
EXPLICIT_CHUNK_SEPARATOR = "\n\n<|LIGHTRAG_PIPELINE_CHUNK|>\n\n"


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def _select_text(doc: Dict[str, Any], prefer_llm_summary: bool) -> str:
    """Return the text to embed for a document. 
    Choose summary text or structured chunks as the insertion payload for one doc."""
    metadata = doc.get("metadata") or {}
    alias_bridge = _build_alias_bridge_chunk(metadata)

    if prefer_llm_summary and doc.get("llm_summary"):
        summary_text = doc["llm_summary"].strip()
        if alias_bridge:
            return f"{summary_text}\n\n{alias_bridge}"
        return summary_text

    ingest_chunks = doc.get("ingest_chunks") or []
    if alias_bridge:
        ingest_chunks = [*ingest_chunks, alias_bridge]

    if ingest_chunks:
        return EXPLICIT_CHUNK_SEPARATOR.join(
            chunk.strip() for chunk in ingest_chunks if chunk and chunk.strip()
        )

    base_text = (doc.get("text", "") or "").strip()
    if alias_bridge:
        return f"{base_text}\n\n{alias_bridge}" if base_text else alias_bridge
    return base_text


def _build_alias_bridge_chunk(metadata: Dict[str, Any]) -> str:
    """Create a compact bilingual alias hint chunk for cross-script retrieval."""
    label_mk = (metadata.get("label_mk") or "").strip()
    label_en = (metadata.get("label_en") or "").strip()
    if not label_mk or not label_en:
        return ""
    if label_mk.casefold() == label_en.casefold():
        return ""
    return (
        "Entity aliases (same entity): "
        f"Macedonian/Cyrillic: {label_mk}; English/Latin: {label_en}."
    )


def _build_batches(
    documents: List[Dict[str, Any]],
    prefer_llm_summary: bool,
) -> Tuple[List[str], List[str], List[str], Dict[str, Dict[str, Any]]]:
    """Return (texts, ids, file_paths, metadata_by_id) lists, filtering out blank documents."""
    texts: List[str] = []
    ids: List[str] = []
    file_paths: List[str] = []
    metadata_by_id: Dict[str, Dict[str, Any]] = {}
    for doc in documents:
        text = _select_text(doc, prefer_llm_summary)
        if text.strip():
            metadata = doc.get("metadata") or {}
            source_url = metadata.get("wikipedia_url") or f"wikidata://{doc['qid']}"
            texts.append(text)
            ids.append(doc["qid"])
            file_paths.append(source_url)
            metadata_by_id[doc["qid"]] = metadata
    return texts, ids, file_paths, metadata_by_id


def _storage_backends_from_env() -> Dict[str, Any]:
    """Mirror the API server's storage backend selection for standalone ingestion."""
    return {
        "kv_storage": os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        "vector_storage": os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
        "graph_storage": os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
        "doc_status_storage": os.getenv(
            "LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"
        ),
    }


def _collect_cross_language_merge_pairs(
    documents: List[Dict[str, Any]],
    canonical_lang: str,
) -> List[Tuple[str, str]]:
    """Build unique (source_entity, target_entity) pairs for mk/en aliases.

    The target is the canonical name chosen by ``canonical_lang``.
    """
    seen: set[Tuple[str, str]] = set()
    pairs: List[Tuple[str, str]] = []

    for doc in documents:
        metadata = doc.get("metadata") or {}
        label_mk = (metadata.get("label_mk") or "").strip()
        label_en = (metadata.get("label_en") or "").strip()
        if not label_mk or not label_en:
            continue
        if label_mk.casefold() == label_en.casefold():
            continue

        if canonical_lang == "en":
            target, source = label_en, label_mk
        else:
            target, source = label_mk, label_en

        pair = (source, target)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    return pairs


async def _merge_cross_language_entities(
    rag: Any,
    documents: List[Dict[str, Any]],
) -> None:
    """Merge mk/en duplicate entity nodes into one canonical node per entity."""
    if os.getenv("LIGHTRAG_MERGE_CROSS_LANGUAGE_ALIASES", "true").lower() != "true":
        return

    canonical_lang = os.getenv("LIGHTRAG_CANONICAL_ENTITY_LANG", "mk").strip().lower()
    if canonical_lang not in {"mk", "en"}:
        canonical_lang = "mk"

    merge_pairs = _collect_cross_language_merge_pairs(documents, canonical_lang)
    if not merge_pairs:
        return

    merged = 0
    skipped = 0

    for source_name, target_name in merge_pairs:
        try:
            source_exists = await rag.chunk_entity_relation_graph.has_node(source_name)
            target_exists = await rag.chunk_entity_relation_graph.has_node(target_name)

            if not source_exists:
                skipped += 1
                continue

            if source_name == target_name:
                skipped += 1
                continue

            target_overrides: Dict[str, Any] = {}
            if not target_exists:
                target_overrides["entity_type"] = "concept"

            await rag.amerge_entities(
                source_entities=[source_name],
                target_entity=target_name,
                target_entity_data=target_overrides or None,
            )
            merged += 1
        except Exception as exc:
            skipped += 1
            print(
                f"  Cross-language merge skipped for '{source_name}' -> '{target_name}': {exc}"
            )

    print(
        "  Cross-language entity merge: "
        f"{merged} merged, {skipped} skipped "
        f"(canonical={canonical_lang})"
    )


async def _persist_doc_metadata(
    rag: Any,
    doc_ids: List[str],
    metadata_by_id: Dict[str, Dict[str, Any]],
) -> None:
    """Attach source metadata to LightRAG doc_status rows after insertion."""
    updates: Dict[str, Dict[str, Any]] = {}
    for doc_id in doc_ids:
        existing = await rag.doc_status.get_by_id(doc_id)
        if not existing:
            continue
        merged_metadata = dict(existing.get("metadata") or {})
        merged_metadata.update(metadata_by_id.get(doc_id) or {})
        updates[doc_id] = {
            "status": existing["status"],
            "content_summary": existing["content_summary"],
            "content_length": existing["content_length"],
            "chunks_count": existing.get("chunks_count", -1),
            "chunks_list": existing.get("chunks_list", []),
            "created_at": existing["created_at"],
            "updated_at": existing["updated_at"],
            "file_path": existing["file_path"],
            "track_id": existing.get("track_id"),
            "error_msg": existing.get("error_msg"),
            "metadata": merged_metadata,
        }
    if updates:
        await rag.doc_status.upsert(updates)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_documents(
    documents: List[Dict[str, Any]],
    working_dir: Optional[Path] = None,
    prefer_llm_summary: bool = False,
    batch_size: int = 4,
    split_by_character: str = DEFAULT_SPLIT_BY_CHARACTER,
    split_by_character_only: bool = False,
    llm_func: Optional[Callable] = None,
    embed_func: Optional[Any] = None,
) -> None:
    """
    Insert documents into LightRAG.

    Args:
        documents:          List of {"qid", "text", "metadata", optionally "llm_summary"}.
        working_dir:        LightRAG storage directory.  Defaults to data/rag_storage.
        prefer_llm_summary: Use ``llm_summary`` instead of ``text`` when available.
        batch_size:         Parallel insert concurrency (max recommended: 10).
        split_by_character: Paragraph boundary used by LightRAG chunking. Default: "\\n\\n".
        split_by_character_only: If True, split only by character boundary and skip token-aware fallback.
        llm_func:           Override default LLM function (gpt_4o_mini_complete).
        embed_func:         Override default embedding function (openai_embed).
    """
    # ── Import guards ─────────────────────────────────────────────────────────
    try:
        from lightrag import LightRAG
    except ImportError:
        print("LightRAG is not installed.  Run: pip install -e .[api]")
        return

    llm_model_name: str = os.getenv("LLM_MODEL", "")
    llm_model_kwargs: Dict[str, Any] = {}

    if llm_func is None or embed_func is None:
        binding = os.getenv("LLM_BINDING", "openai").lower()
        if binding == "ollama":
            try:
                from functools import partial
                from lightrag.llm.ollama import ollama_model_complete, ollama_embed
                from lightrag.utils import EmbeddingFunc

                async def _ollama_llm_wrapper(*args, **kwargs):
                    # LightRAG may pass model in kwargs; ollama_model_complete already
                    # resolves model from global config, so remove duplicate.
                    kwargs.pop("model", None)
                    return await ollama_model_complete(*args, **kwargs)

                host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
                embed_host = os.getenv("EMBEDDING_BINDING_HOST", host)
                if llm_func is None:
                    llm_func = _ollama_llm_wrapper
                    llm_model_name = os.getenv("LLM_MODEL", "qwen2.5:7b")
                    llm_model_kwargs = {
                        "host": host,
                        "options": {"num_ctx": int(os.getenv("OLLAMA_LLM_NUM_CTX", "8192"))},
                        "timeout": int(os.getenv("LLM_TIMEOUT", "600")),
                    }
                if embed_func is None:
                    embedding_model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
                    embed_func = EmbeddingFunc(
                        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
                        max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
                        model_name=embedding_model,
                        func=partial(
                            ollama_embed.func,
                            embed_model=embedding_model,
                            host=embed_host,
                        ),
                    )
            except ImportError as exc:
                print(f"Cannot import Ollama LLM bindings: {exc}")
                return
        else:
            api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("LLM_BINDING_API_KEY / OPENAI_API_KEY is not set.  Skipping ingestion.")
                return
            try:
                from functools import partial
                from lightrag.llm.openai import openai_complete_if_cache, openai_embed
                from lightrag.utils import EmbeddingFunc

                model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
                base_url = os.getenv("LLM_BINDING_HOST") or None
                embedding_base_url = os.getenv("EMBEDDING_BINDING_HOST") or base_url
                embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
                embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
                max_embed_tokens = int(os.getenv("MAX_EMBED_TOKENS", "8192"))
                timeout = int(os.getenv("LLM_TIMEOUT", "600"))

                if llm_func is None:
                    async def _openai_llm_wrapper(*args, **kwargs):
                        kwargs.pop("hashing_kv", None)
                        return await openai_complete_if_cache(
                            model,
                            *args,
                            api_key=api_key,
                            base_url=base_url,
                            timeout=timeout,
                            **kwargs,
                        )

                    llm_func = _openai_llm_wrapper
                    llm_model_name = model
                if embed_func is None:
                    embed_func = EmbeddingFunc(
                        embedding_dim=embedding_dim,
                        max_token_size=max_embed_tokens,
                        model_name=embedding_model,
                        func=partial(
                            openai_embed.func,
                            model=embedding_model,
                            api_key=api_key,
                            base_url=embedding_base_url,
                        ),
                    )
            except ImportError as exc:
                print(f"Cannot import OpenAI LLM bindings: {exc}")
                return

    # ── Storage setup ─────────────────────────────────────────────────────────
    storage = working_dir or DEFAULT_WORKING_DIR
    storage.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(storage),
        llm_model_func=llm_func,
        llm_model_name=llm_model_name,
        llm_model_kwargs=llm_model_kwargs,
        embedding_func=embed_func,
        max_parallel_insert=batch_size,
        **_storage_backends_from_env(),
    )
    await rag.initialize_storages()

    # ── Build text / id lists ─────────────────────────────────────────────────
    texts, ids, file_paths, metadata_by_id = _build_batches(documents, prefer_llm_summary)

    if not texts:
        print("No non-empty documents to ingest.")
        await rag.finalize_storages()
        return

    mode = "llm_summary" if prefer_llm_summary else "structured_text"
    print(
        f"  Inserting {len(texts)} documents  (text mode: {mode}, "
        f"split_by={split_by_character!r}) ..."
    )

    actual_split_by = split_by_character
    actual_split_by_character_only = split_by_character_only
    if not prefer_llm_summary:
        actual_split_by = EXPLICIT_CHUNK_SEPARATOR
        # Phase 4 already prepares bounded ingest chunks, so keep them intact.
        actual_split_by_character_only = True

    await rag.ainsert(
        texts,
        split_by_character=actual_split_by,
        split_by_character_only=actual_split_by_character_only,
        ids=ids,
        file_paths=file_paths,
    )
    await _persist_doc_metadata(rag, ids, metadata_by_id)
    await _merge_cross_language_entities(rag, documents)
    await rag.finalize_storages()

    print(f"  Ingestion complete.  Storage: {storage}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    async def _main() -> None:
        src = DATA_DIR / "normalized_documents.json"
        if not src.exists():
            print(f"Run normalization.py first to generate {src}.")
            return
        docs = json.loads(src.read_text(encoding="utf-8"))
        print(f"Loaded {len(docs)} documents for ingestion.")
        split_by = os.getenv("LIGHTRAG_INGEST_SPLIT_BY_CHARACTER", DEFAULT_SPLIT_BY_CHARACTER)
        split_only = os.getenv("LIGHTRAG_INGEST_SPLIT_BY_CHARACTER_ONLY", "false").lower() == "true"
        await ingest_documents(docs, split_by_character=split_by, split_by_character_only=split_only)

    asyncio.run(_main())
