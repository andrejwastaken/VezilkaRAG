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
) -> Tuple[List[str], List[str]]:
    """Return (texts, ids) lists, filtering out blank documents."""
    texts: List[str] = []
    ids: List[str] = []
    for doc in documents:
        text = _select_text(doc, prefer_llm_summary)
        if text.strip():
            texts.append(text)
            ids.append(doc["qid"])
    return texts, ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_documents(
    documents: List[Dict[str, Any]],
    working_dir: Optional[Path] = None,
    prefer_llm_summary: bool = False,
    batch_size: int = 4,
    llm_func: Optional[Callable] = None,
    embed_func: Optional[Callable] = None,
) -> None:
    """
    Insert documents into LightRAG.

    Args:
        documents:          List of {"qid", "text", "metadata", optionally "llm_summary"}.
        working_dir:        LightRAG storage directory.  Defaults to data/rag_storage.
        prefer_llm_summary: Use ``llm_summary`` instead of ``text`` when available.
        batch_size:         Parallel insert concurrency (max recommended: 10).
        llm_func:           Override default LLM function (gpt_4o_mini_complete).
        embed_func:         Override default embedding function (openai_embed).
    """
    # ── Import guards ─────────────────────────────────────────────────────────
    try:
        from lightrag import LightRAG
    except ImportError:
        print("LightRAG is not installed.  Run: pip install -e .[api]")
        return

    if llm_func is None or embed_func is None:
        binding = os.getenv("LLM_BINDING", "openai").lower()
        if binding == "ollama":
            try:
                from functools import partial
                from lightrag.llm.ollama import ollama_model_complete, ollama_embed
                from lightrag.utils import EmbeddingFunc
                host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
                embed_host = os.getenv("EMBEDDING_BINDING_HOST", host)
                if llm_func is None:
                    llm_func = partial(
                        ollama_model_complete,
                        hashing_kv=None,
                        host=host,
                        model=os.getenv("LLM_MODEL", "qwen2.5:32b"),
                        options={"num_ctx": int(os.getenv("LLM_CTX", "32768"))},
                    )
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
        embedding_func=embed_func,
        max_parallel_insert=batch_size,
    )
    await rag.initialize_storages()

    # ── Build text / id lists ─────────────────────────────────────────────────
    texts, ids = _build_batches(documents, prefer_llm_summary)

    if not texts:
        print("No non-empty documents to ingest.")
        await rag.finalize_storages()
        return

    mode = "llm_summary" if prefer_llm_summary else "structured_text"
    print(f"  Inserting {len(texts)} documents  (text mode: {mode}) …")

    await rag.ainsert(texts, ids=ids)
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
        await ingest_documents(docs)

    asyncio.run(_main())
