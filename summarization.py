"""
Phase 5 – Optional LLM Summarization
=======================================

Optionally compresses a normalized document into a compact narrative summary.
This phase is NOT required for ingestion when chunked deterministic text from
Phase 4 is used directly.

Stored separately from the structured paragraph:
  - ``text``        →  deterministic factual text (Phase 4)
  - ``llm_summary`` →  LLM narrative (Phase 5, optional)

Best practice with chunked ingestion:
    - Use ``text`` for retrieval quality and lower hallucination risk
    - Use ``llm_summary`` mainly for UI display or quick previews
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a cultural knowledge editor writing compact encyclopedia-style summaries. "
    "Your summaries are factual, third-person, and encyclopedic. "
    "You never add information that is not present in the provided facts."
)

SUMMARY_INPUT_MAX_CHARS = int(os.getenv("LIGHTRAG_SUMMARY_INPUT_MAX_CHARS", "9000"))

USER_TEMPLATE = """\
Below are structured facts about a cultural entity from North Macedonia.
Write a concise summary paragraph (3–5 sentences) in English.

RULES:
- Use ONLY the facts listed below. Do NOT hallucinate or infer unstated details.
- Write in the third person.
- Be factual and encyclopedic – no opinions or evaluations.
- If the facts are insufficient for a meaningful paragraph, write a single sentence.

FACTS:
{facts}

SUMMARY:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[TRUNCATED FOR SUMMARIZATION INPUT]"


def _doc_to_facts(doc: Dict[str, Any]) -> str:
    """Format one normalized document as bounded facts for summarization."""
    meta = doc.get("metadata", {})
    label = meta.get("label_en") or meta.get("label_mk") or doc["qid"]
    body = doc.get("text", "")
    wiki_chunk_count = meta.get("wikipedia_chunk_count", 0)
    wiki_lang = meta.get("wikipedia_lang") or "unknown"
    body = _truncate(body, SUMMARY_INPUT_MAX_CHARS)
    return (
        f"Entity: {label}\n"
        f"QID: {doc['qid']}\n"
        f"Wikipedia language: {wiki_lang}\n"
        f"Wikipedia chunks: {wiki_chunk_count}\n\n"
        f"{body}"
    )


async def _call_llm(prompt: str, llm_func: Callable) -> Optional[str]:
    """Call the LLM with a single prompt string; handle both plain-text and message responses."""
    combined_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    result = await llm_func(combined_prompt)
    if isinstance(result, str):
        return result.strip() or None
    # Handle ChatCompletion-style objects
    if hasattr(result, "choices"):
        content = result.choices[0].message.content
        return content.strip() if content else None
    return str(result).strip() or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def summarize_document(
    doc: Dict[str, Any],
    llm_func: Callable,
) -> Optional[str]:
    """
    Generate an LLM summary for a single document.

    Args:
        doc:      Normalized document dict (must have "text" and "qid").
        llm_func: Async callable that accepts a prompt string and returns a string
                  (e.g. ``lightrag.llm.openai.gpt_4o_mini_complete``).

    Returns:
        Summary string, or None on failure.
    """
    facts = _doc_to_facts(doc)
    if not facts.strip():
        return None
    prompt = USER_TEMPLATE.format(facts=facts)
    try:
        return await _call_llm(prompt, llm_func)
    except Exception as exc:
        print(f"  Summarization error for {doc['qid']}: {exc}")
        return None


async def summarize_all(
    documents: List[Dict[str, Any]],
    llm_func: Callable,
    max_concurrent: int = 3,
) -> List[Dict[str, Any]]:
    """
    Summarize all documents concurrently (max_concurrent at a time).

    Adds / updates the ``llm_summary`` key on each document dict for which a
    summary was successfully generated.  Documents without a summary are left
    unchanged.

    Args:
        documents:      List of normalized document dicts.
        llm_func:       Async LLM callable.
        max_concurrent: Max simultaneous LLM requests (be polite to rate limits).

    Returns:
        The same list, mutated in-place.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(documents)

    async def _bounded(idx: int, doc: Dict[str, Any]) -> None:
        async with semaphore:
            label = doc.get("metadata", {}).get("label_en") or doc["qid"]
            print(f"  [{idx:>4}/{total}] Summarizing {label[:50]} …", end=" ", flush=True)
            summary = await summarize_document(doc, llm_func)
            if summary:
                doc["llm_summary"] = summary
                print("OK")
            else:
                print("(no result)")

    await asyncio.gather(*[_bounded(i + 1, d) for i, d in enumerate(documents)])
    summarized = sum(1 for d in documents if d.get("llm_summary"))
    print(f"\nSummarized {summarized}/{total} documents.")
    return documents


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import pathlib

    async def _main() -> None:
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set.")
            return
        try:
            from lightrag.llm.openai import gpt_4o_mini_complete
        except ImportError:
            print("LightRAG is not installed.")
            return

        src = pathlib.Path("data/normalized_documents.json")
        if not src.exists():
            print("Run normalization.py first.")
            return

        docs = json.loads(src.read_text(encoding="utf-8"))
        sample = docs[:5]
        sample = await summarize_all(sample, gpt_4o_mini_complete)

        print("\n── Summaries ────────────────────────────────────────────")
        for d in sample:
            print(f"\n{d['qid']}  {d.get('metadata', {}).get('label_en', '')}")
            print(d.get("llm_summary", "(none)"))

    asyncio.run(_main())
