"""
Macedonian Cultural Knowledge Pipeline
======================================

Orchestrates all six phases of the Wikidata-driven ingestion pipeline:

  Phase 1 – Wikidata Discovery      (wikidata_discovery.py)
  Phase 2 – Detail Expansion        (wikidata_detail.py)
  Phase 3 – Wikipedia Article Fetch (wikipedia_fetch.py)
  Phase 4 – Normalization           (normalization.py)   [default ingestion path]
  Phase 5 – LLM Summarization       (summarization.py)   [optional compression]
  Phase 6 – LightRAG Ingestion      (ingestion.py)

Recommended usage:
  - Use Phase 4 output for real ingestion and querying.
  - Use Phase 5 only when you explicitly want shorter summaries or a speed/cost
    fallback.

Intermediate results are persisted as JSON under data/ so any phase can be
resumed without re-running earlier ones.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Intermediate artefact paths
# ---------------------------------------------------------------------------

REPO_ROOT             = Path(__file__).resolve().parent.parent
DATA_DIR              = REPO_ROOT / "data"
DISCOVERED_QIDS_FILE  = DATA_DIR / "discovered_qids.json"
ENTITY_DETAILS_FILE   = DATA_DIR / "entity_details.json"
NORMALIZED_DOCS_FILE  = DATA_DIR / "normalized_documents.json"


# ---------------------------------------------------------------------------
# JSON persistence helpers
# ---------------------------------------------------------------------------

def _save(path: Path, data: Any) -> None:
    # Persist a JSON artifact to disk, creating parent directories when needed.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load(path: Path) -> Any:
    # Load and parse a JSON artifact from disk.
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def phase1_discovery(limit: int, categories: list[str] | None = None) -> list:
    # Run phase 1 discovery and save discovered QIDs.
    from wikidata_discovery import discover_entities
    print("\n=== Phase 1: Wikidata Discovery ===")
    qids = discover_entities(limit_per_category=limit, category_names=categories)
    _save(DISCOVERED_QIDS_FILE, qids)
    print(f"Discovered {len(qids)} unique QIDs → {DISCOVERED_QIDS_FILE}")
    return qids


def phase2_detail(qids: list) -> dict:
    # Run phase 2 detail expansion and save structured entity payloads.
    from wikidata_detail import fetch_all_entities
    print("\n=== Phase 2: Detail Expansion ===")
    entities = fetch_all_entities(qids)
    _save(ENTITY_DETAILS_FILE, entities)
    print(f"Fetched details for {len(entities)} entities → {ENTITY_DETAILS_FILE}")
    return entities


def phase3_wikipedia(entities: dict) -> dict:
    # Run phase 3 Wikipedia enrichment and persist updated entities.
    from wikipedia_fetch import enrich_with_wikipedia
    print("\n=== Phase 3: Wikipedia Article Fetch ===")
    entities = enrich_with_wikipedia(entities)
    _save(ENTITY_DETAILS_FILE, entities)
    return entities


def phase4_normalize(entities: dict, lang: str = "en") -> list:
    # Run phase 4 normalization to build deterministic ingestion documents.
    from normalization import normalize_all
    print(f"\n=== Phase 4: Normalization (lang={lang}) ===")
    docs = normalize_all(entities, lang=lang)
    _save(NORMALIZED_DOCS_FILE, docs)
    print(f"Normalized {len(docs)} documents → {NORMALIZED_DOCS_FILE}")
    return docs


async def phase5_summarize(docs: list, max_concurrent: int = 3) -> list:
    # Run optional phase 5 summarization using the configured LLM binding.
    print("\n=== Phase 5: LLM Summarization ===")
    binding = os.getenv("LLM_BINDING", "openai").lower()
    llm_func = None
    if binding == "ollama":
        try:
            import ollama

            host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
            model = os.getenv("LLM_MODEL", "qwen2.5:7b")
            num_ctx = int(os.getenv("OLLAMA_LLM_NUM_CTX", "8192"))

            async def _ollama_prompt_llm(prompt: str) -> str:
                client = ollama.AsyncClient(host=host, timeout=int(os.getenv("LLM_TIMEOUT", "600")))
                try:
                    response = await client.chat(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_ctx": num_ctx},
                    )
                    return (response.get("message") or {}).get("content", "").strip()
                finally:
                    await client._client.aclose()

            llm_func = _ollama_prompt_llm
        except ImportError:
            print("LightRAG Ollama bindings not found – skipping summarization.")
            return docs
    else:
        api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("LLM_BINDING_API_KEY / OPENAI_API_KEY not set – skipping summarization.")
            return docs
        try:
            from lightrag.llm.openai import openai_complete_if_cache

            model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
            base_url = os.getenv("LLM_BINDING_HOST") or None
            timeout = int(os.getenv("LLM_TIMEOUT", "600"))

            async def _openai_prompt_llm(prompt: str) -> str:
                return await openai_complete_if_cache(
                    model,
                    prompt,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )

            llm_func = _openai_prompt_llm
        except ImportError:
            print("LightRAG not installed – skipping summarization.")
            return docs

    from summarization import summarize_all
    docs = await summarize_all(docs, llm_func, max_concurrent=max_concurrent)
    _save(NORMALIZED_DOCS_FILE, docs)
    summarized = sum(1 for d in docs if d.get("llm_summary"))
    print(f"Summaries generated: {summarized}/{len(docs)}")
    return docs


async def phase6_ingest(docs: list, prefer_llm_summary: bool) -> None:
    # Run phase 6 ingestion into LightRAG storages.
    from ingestion import ingest_documents
    print("\n=== Phase 6: LightRAG Ingestion ===")
    await ingest_documents(docs, prefer_llm_summary=prefer_llm_summary)


# ---------------------------------------------------------------------------
# Resumable loader helpers
# ---------------------------------------------------------------------------

def _require(path: Path, label: str) -> Any:
    # Ensure a prerequisite artifact exists, otherwise raise a resume hint.
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found – run the pipeline from an earlier phase first.\n"
            f"Hint: python pipeline.py --from-phase 1"
        )
    data = _load(path)
    print(f"Loaded {label} from {path}")
    return data


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    # Orchestrate all phases with resume support and optional summarization.
    start = args.from_phase
    run_summarize = args.run_summarize
    limit = args.limit
    categories = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None
    lang = args.lang

    print("Macedonian Cultural Knowledge Pipeline")
    print(f"  Start phase : {start}")
    print(f"  Limit/cat   : {limit}")
    print(f"  Categories  : {', '.join(categories) if categories else 'all defaults'}")
    print(f"  Text lang   : {lang}")
    print(
        f"  Summarize   : "
        f"{'yes (Phase 5, faster/lower-detail fallback)' if run_summarize else 'no (Phase 4 only, recommended)'}"
    )

    # ── Phase 1: Discovery ────────────────────────────────────────────────────
    if start <= 1:
        qids = phase1_discovery(limit, categories)
    else:
        qids = _require(DISCOVERED_QIDS_FILE, f"{len(_load(DISCOVERED_QIDS_FILE))} QIDs")

    # ── Phase 2: Detail expansion ─────────────────────────────────────────────
    if start <= 2:
        entities = phase2_detail(qids)
    else:
        entities = _require(ENTITY_DETAILS_FILE, "entity details")

    # ── Phase 3: Wikipedia fetch ──────────────────────────────────────────────
    if start <= 3:
        entities = phase3_wikipedia(entities)
    elif start > 3 and isinstance(entities, dict):
        pass  # already loaded with Wikipedia extracts included

    # ── Phase 4: Normalization ────────────────────────────────────────────────
    if start <= 4:
        docs = phase4_normalize(entities, lang=lang)
    else:
        docs = _require(NORMALIZED_DOCS_FILE, "normalized documents")

    # ── Phase 5: LLM Summarization (optional) ────────────────────────────────
    if run_summarize and start <= 5:
        docs = await phase5_summarize(docs)

    # ── Phase 6: LightRAG Ingestion ───────────────────────────────────────────
    if start <= 6:
        await phase6_ingest(docs, prefer_llm_summary=run_summarize)

    print("\nPipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Macedonian Cultural Knowledge Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from-phase", type=int, default=1, metavar="N",
        help="Resume from phase N (1–6).  Default: 1 (full run).",
    )
    parser.add_argument(
        "--run-summarize", action="store_true",
        help="Run Phase 5 (LLM summarization). Default is OFF to reduce cost.",
    )
    parser.add_argument(
        "--limit", type=int, default=1, metavar="N",
        help="Max entities per category during discovery. Default: 1.",
    )
    parser.add_argument(
        "--categories", type=str, default="",
        help=(
            "Comma-separated category names for a smaller discovery set, "
            "for example: 'Artists,Books,Towns'. Default: all categories."
        ),
    )
    parser.add_argument(
        "--lang", choices=["en", "mk"], default="en",
        help="Language for fact-sentence templates in Phase 4.  Default: en.",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
