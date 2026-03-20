"""
Macedonian Cultural Knowledge Pipeline
========================================

Orchestrates all six phases of the Wikidata-driven ingestion pipeline:

  Phase 1 – Wikidata Discovery      (wikidata_discovery.py)
  Phase 2 – Detail Expansion        (wikidata_detail.py)
  Phase 3 – Wikipedia Article Fetch (wikipedia_fetch.py)
  Phase 4 – Normalization           (normalization.py)
  Phase 5 – LLM Summarization       (summarization.py)   [optional]
  Phase 6 – LightRAG Ingestion      (ingestion.py)

Intermediate results are persisted as JSON under data/ so any phase can
be resumed without re-running earlier ones.

Usage examples
--------------
  python pipeline.py                       # full run, all phases
    python pipeline.py --run-summarize       # include Phase 5 (LLM cost)
  python pipeline.py --from-phase 3        # resume from Wikipedia fetch
  python pipeline.py --limit 100           # faster test run
  python pipeline.py --from-phase 6        # re-ingest existing documents
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def phase1_discovery(limit: int) -> list:
    from wikidata_discovery import discover_entities
    print("\n=== Phase 1: Wikidata Discovery ===")
    qids = discover_entities(limit_per_category=limit)
    _save(DISCOVERED_QIDS_FILE, qids)
    print(f"Discovered {len(qids)} unique QIDs → {DISCOVERED_QIDS_FILE}")
    return qids


def phase2_detail(qids: list) -> dict:
    from wikidata_detail import fetch_all_entities
    print("\n=== Phase 2: Detail Expansion ===")
    entities = fetch_all_entities(qids)
    _save(ENTITY_DETAILS_FILE, entities)
    print(f"Fetched details for {len(entities)} entities → {ENTITY_DETAILS_FILE}")
    return entities


def phase3_wikipedia(entities: dict) -> dict:
    from wikipedia_fetch import enrich_with_wikipedia
    print("\n=== Phase 3: Wikipedia Article Fetch ===")
    entities = enrich_with_wikipedia(entities)
    _save(ENTITY_DETAILS_FILE, entities)
    return entities


def phase4_normalize(entities: dict, lang: str = "en") -> list:
    from normalization import normalize_all
    print(f"\n=== Phase 4: Normalization (lang={lang}) ===")
    docs = normalize_all(entities, lang=lang)
    _save(NORMALIZED_DOCS_FILE, docs)
    print(f"Normalized {len(docs)} documents → {NORMALIZED_DOCS_FILE}")
    return docs


async def phase5_summarize(docs: list, max_concurrent: int = 3) -> list:
    print("\n=== Phase 5: LLM Summarization ===")
    binding = os.getenv("LLM_BINDING", "openai").lower()
    llm_func = None
    if binding == "ollama":
        try:
            from functools import partial
            from lightrag.llm.ollama import ollama_model_complete
            host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
            llm_func = partial(
                ollama_model_complete,
                hashing_kv=None,
                host=host,
                model=os.getenv("LLM_MODEL", "qwen3.5:7b"),
                options={"num_ctx": int(os.getenv("LLM_CTX", "32768"))},
            )
        except ImportError:
            print("LightRAG Ollama bindings not found – skipping summarization.")
            return docs
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set – skipping summarization.")
            return docs
        try:
            from lightrag.llm.openai import gpt_4o_mini_complete
            llm_func = gpt_4o_mini_complete
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
    from ingestion import ingest_documents
    print("\n=== Phase 6: LightRAG Ingestion ===")
    await ingest_documents(docs, prefer_llm_summary=prefer_llm_summary)


# ---------------------------------------------------------------------------
# Resumable loader helpers
# ---------------------------------------------------------------------------

def _require(path: Path, label: str) -> Any:
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
    start = args.from_phase
    run_summarize = args.run_summarize
    limit = args.limit

    lang = args.lang

    print("Macedonian Cultural Knowledge Pipeline")
    print(f"  Start phase : {start}")
    print(f"  Limit/cat   : {limit}")
    print(f"  Text lang   : {lang}")
    print(f"  Summarize   : {'yes (Phase 5)' if run_summarize else 'no (default)'}")

    # ── Phase 1: Discovery ────────────────────────────────────────────────────
    if start <= 1:
        qids = phase1_discovery(limit)
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
        "--limit", type=int, default=500, metavar="N",
        help="Max entities per category during discovery.  Default: 500.",
    )
    parser.add_argument(
        "--lang", choices=["en", "mk"], default="en",
        help="Language for fact-sentence templates in Phase 4.  Default: en.",
    )
    args = parser.parse_args()
    asyncio.run(main(args))