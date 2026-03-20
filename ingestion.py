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

DEFAULT_WORKING_DIR = Path(__file__).parent / "data" / "rag_storage"
DEFAULT_SPLIT_BY_CHARACTER = "\n\n"


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def _select_text(doc: Dict[str, Any], prefer_llm_summary: bool) -> str:
    """Return the text to embed for a document."""
    if prefer_llm_summary and doc.get("llm_summary"):
        return doc["llm_summary"]
    return doc.get("text", "")


def _build_batches(
    documents: List[Dict[str, Any]],
    prefer_llm_summary: bool,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (texts, ids, file_paths) lists, filtering out blank documents."""
    texts: List[str] = []
    ids: List[str] = []
    file_paths: List[str] = []
    for doc in documents:
        text = _select_text(doc, prefer_llm_summary)
        if text.strip():
            metadata = doc.get("metadata") or {}
            source_url = metadata.get("wikipedia_url") or f"wikidata://{doc['qid']}"
            texts.append(text)
            ids.append(doc["qid"])
            file_paths.append(source_url)
    return texts, ids, file_paths


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
                    embed_func = EmbeddingFunc(
                        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
                        max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
                        func=partial(
                            ollama_embed.func,
                            embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                            host=embed_host,
                        ),
                    )
            except ImportError as exc:
                print(f"Cannot import Ollama LLM bindings: {exc}")
                return
        else:
            if not os.getenv("OPENAI_API_KEY"):
                print("OPENAI_API_KEY is not set.  Skipping ingestion.")
                return
            try:
                from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
                if llm_func is None:
                    llm_func = gpt_4o_mini_complete
                if embed_func is None:
                    embed_func = openai_embed
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
    )
    await rag.initialize_storages()

    # ── Build text / id lists ─────────────────────────────────────────────────
    texts, ids, file_paths = _build_batches(documents, prefer_llm_summary)

    if not texts:
        print("No non-empty documents to ingest.")
        await rag.finalize_storages()
        return

    mode = "llm_summary" if prefer_llm_summary else "structured_text"
    print(
        f"  Inserting {len(texts)} documents  (text mode: {mode}, "
        f"split_by={split_by_character!r}) ..."
    )

    await rag.ainsert(
        texts,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
        file_paths=file_paths,
    )
    await rag.finalize_storages()

    print(f"  Ingestion complete.  Storage: {storage}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import pathlib

    async def _main() -> None:
        src = pathlib.Path("data/normalized_documents.json")
        if not src.exists():
            print("Run normalization.py first to generate data/normalized_documents.json.")
            return
        docs = json.loads(src.read_text(encoding="utf-8"))
        print(f"Loaded {len(docs)} documents for ingestion.")
        split_by = os.getenv("LIGHTRAG_INGEST_SPLIT_BY_CHARACTER", DEFAULT_SPLIT_BY_CHARACTER)
        split_only = os.getenv("LIGHTRAG_INGEST_SPLIT_BY_CHARACTER_ONLY", "false").lower() == "true"
        await ingest_documents(docs, split_by_character=split_by, split_by_character_only=split_only)

    asyncio.run(_main())
