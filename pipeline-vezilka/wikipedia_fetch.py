"""
Phase 3 – Wikipedia Article Fetch
===================================

For each entity that has a Macedonian (mk) or English (en) Wikipedia sitelink
(discovered in Phase 2), fetches the article summary via the REST API.

Macedonian is preferred over English because it provides native-language
paragraphs which dramatically improve RAG quality for Macedonian queries.

REST API docs: https://en.wikipedia.org/api/rest_v1/#/Page_content/get_page_summary__title_
"""
from __future__ import annotations

import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

import requests

MK_WIKI_SUMMARY = "https://mk.wikipedia.org/api/rest_v1/page/summary/{title}"
EN_WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
MK_WIKI_API = "https://mk.wikipedia.org/w/api.php"
EN_WIKI_API = "https://en.wikipedia.org/w/api.php"
MK_WIKI_PAGE = "https://mk.wikipedia.org/wiki/{title}"
EN_WIKI_PAGE = "https://en.wikipedia.org/wiki/{title}"

# Full articles can be very large; cap length for practical ingestion volume.
MAX_FULLTEXT_CHARS = 30000
DEFAULT_CHUNK_SIZE_TOKENS = 450
DEFAULT_CHUNK_OVERLAP_TOKENS = 60

CHUNK_SIZE_TOKENS = int(os.getenv("LIGHTRAG_WIKI_CHUNK_TOKENS", str(DEFAULT_CHUNK_SIZE_TOKENS)))
CHUNK_OVERLAP_TOKENS = int(
    os.getenv("LIGHTRAG_WIKI_CHUNK_OVERLAP_TOKENS", str(DEFAULT_CHUNK_OVERLAP_TOKENS))
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

REQUEST_DELAY = 0.5  # seconds between requests  (polite crawling)
HEADERS = {"User-Agent": "MacedonianCulturePipeline/1.0 (research@example.com)"}


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def fetch_summary(title: str, lang: str = "mk") -> Optional[str]:
    """
    Fetch the extract (plain-text summary) for a Wikipedia article.

    Args:
        title: Article title as it appears in the sitelinks dict.
        lang:  "mk" or "en".

    Returns:
        Plain-text extract string, or None if unavailable.
    """
    template = MK_WIKI_SUMMARY if lang == "mk" else EN_WIKI_SUMMARY
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = template.format(title=encoded)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            extract = resp.json().get("extract", "").strip()
            return extract or None
        return None
    except Exception:
        return None


def _fetch_full_text(title: str, lang: str = "mk") -> Optional[dict]:
    """Fetch full plaintext article content via MediaWiki action API."""
    api_url = MK_WIKI_API if lang == "mk" else EN_WIKI_API
    try:
        resp = requests.get(
            api_url,
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": 1,
                "exsectionformat": "plain",
                "redirects": 1,
                "titles": title,
            },
            headers=HEADERS,
            timeout=20,
        )
        if resp.status_code != 200:
            return None

        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            if page.get("missing") is not None:
                return None
            raw = (page.get("extract") or "").strip()
            if not raw:
                return None
            return {
                "title": page.get("title") or title,
                "pageid": page.get("pageid"),
                "full_text": raw[:MAX_FULLTEXT_CHARS],
            }
        return None
    except Exception:
        return None


def _token_count(text: str) -> int:
    """Count tokens with tiktoken when available, otherwise use a regex fallback."""
    try:
        import tiktoken  # type: ignore

        model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback approximates tokens as words/punctuation groups.
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _split_long_unit(unit: str, chunk_size_tokens: int) -> list[str]:
    """Split an oversized unit into smaller sentence/word chunks."""
    out: list[str] = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", unit) if s.strip()]
    for sentence in sentences:
        if _token_count(sentence) <= chunk_size_tokens:
            out.append(sentence)
            continue

        words = sentence.split()
        current_words: list[str] = []
        for word in words:
            candidate = " ".join(current_words + [word]).strip()
            if current_words and _token_count(candidate) > chunk_size_tokens:
                out.append(" ".join(current_words).strip())
                current_words = [word]
            else:
                current_words.append(word)
        if current_words:
            out.append(" ".join(current_words).strip())
    return [x for x in out if x]


def _build_overlap_tail(parts: list[str], overlap_tokens: int) -> list[str]:
    """Keep trailing parts up to overlap_tokens for contextual overlap."""
    if overlap_tokens <= 0 or not parts:
        return []

    tail: list[str] = []
    total = 0
    for part in reversed(parts):
        tail.insert(0, part)
        total += _token_count(part)
        if total >= overlap_tokens:
            break
    return tail


def _chunk_text_for_lightrag(
    text: str,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into token-aware chunks with contextual overlap."""
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not cleaned:
        return []
    if _token_count(cleaned) <= chunk_size_tokens:
        return [cleaned]

    units = [u.strip() for u in re.split(r"\n{2,}", cleaned) if u.strip()]
    if not units:
        units = [cleaned]

    expanded_units: list[str] = []
    for unit in units:
        if _token_count(unit) <= chunk_size_tokens:
            expanded_units.append(unit)
        else:
            expanded_units.extend(_split_long_unit(unit, chunk_size_tokens))

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for unit in expanded_units:
        unit_tokens = _token_count(unit)
        if not current_parts:
            current_parts = [unit]
            current_tokens = unit_tokens
            continue

        if current_tokens + unit_tokens <= chunk_size_tokens:
            current_parts.append(unit)
            current_tokens += unit_tokens
            continue

        chunks.append("\n\n".join(current_parts).strip())

        overlap_parts = _build_overlap_tail(current_parts, overlap_tokens)
        current_parts = overlap_parts + [unit]
        while len(current_parts) > 1 and _token_count("\n\n".join(current_parts)) > chunk_size_tokens:
            current_parts.pop(0)
        current_tokens = _token_count("\n\n".join(current_parts))

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [c for c in chunks if c]


def fetch_article_data(title: str, lang: str = "mk") -> Optional[Dict[str, Any]]:
    """
    Fetch richer article data for ingestion:
    - summary extract (REST API)
    - full plaintext content (MediaWiki action API)
    - basic metadata (title/url/pageid/lang)
    """
    summary = fetch_summary(title, lang=lang)
    full = _fetch_full_text(title, lang=lang)

    if not summary and not full:
        return None

    resolved_title = (full or {}).get("title") or title
    encoded_title = urllib.parse.quote(resolved_title.replace(" ", "_"), safe="")
    page_template = MK_WIKI_PAGE if lang == "mk" else EN_WIKI_PAGE

    full_text = (full or {}).get("full_text")
    chunks = _chunk_text_for_lightrag(full_text) if full_text else []

    return {
        "summary": summary,
        "full_text": full_text,
        "chunks": chunks,
        "title": resolved_title,
        "pageid": (full or {}).get("pageid"),
        "url": page_template.format(title=encoded_title),
        "lang": lang,
    }


# ---------------------------------------------------------------------------
# Enrichment loop
# ---------------------------------------------------------------------------

def enrich_with_wikipedia(
    entities: Dict[str, dict],
    delay: float = REQUEST_DELAY,
) -> Dict[str, dict]:
    """
    Adds richer Wikipedia fields to each entity that has a Wikipedia article.
    Macedonian is tried first; English is used as a fallback.

    Mutates *entities* in-place and returns it.
    """
    total = len(entities)
    found = 0

    for idx, (qid, ent) in enumerate(entities.items(), 1):
        sitelinks = ent.get("sitelinks", {})
        mk_title: Optional[str] = sitelinks.get("mkwiki")
        en_title: Optional[str] = sitelinks.get("enwiki")

        article: Optional[Dict[str, Any]] = None
        source: Optional[str] = None

        if mk_title:
            article = fetch_article_data(mk_title, lang="mk")
            if article:
                source = "mk.wikipedia"

        if not article and en_title:
            article = fetch_article_data(en_title, lang="en")
            if article:
                source = "en.wikipedia"

        if article:
            summary = article.get("summary")
            full_text = article.get("full_text")
            chunks = article.get("chunks") or []

            # Keep compatibility with existing downstream code while enriching data.
            ent["wikipedia_extract"] = summary or full_text or ""
            ent["wikipedia_full_text"] = full_text
            ent["wikipedia_chunks"] = chunks
            ent["wikipedia_chunk_size"] = CHUNK_SIZE_TOKENS
            ent["wikipedia_chunk_overlap"] = CHUNK_OVERLAP_TOKENS
            ent["wikipedia_source"] = source
            ent["wikipedia_title"] = article.get("title")
            ent["wikipedia_url"] = article.get("url")
            ent["wikipedia_lang"] = article.get("lang")
            ent["wikipedia_pageid"] = article.get("pageid")

            found += 1
            label = ent.get("label_en") or ent.get("label_mk") or qid
            content_len = len(full_text or summary or "")
            chunk_count = len(chunks)
            print(
                f"  [{idx:>4}/{total}] {qid}  {label[:40]:<40}  "
                f"{source}  ({content_len} chars, {chunk_count} chunks)"
            )
        else:
            pass  # silent skip – keeps output manageable for large runs

        time.sleep(delay)

    print(f"\nWikipedia articles found: {found}/{total}")
    return entities


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    path = DATA_DIR / "entity_details.json"
    if not path.exists():
        print(f"Run wikidata_detail.py first to generate {path}.")
    else:
        entities = json.loads(path.read_text(encoding="utf-8"))
        entities = enrich_with_wikipedia(entities)
        path.write_text(json.dumps(entities, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Updated {len(entities)} entities → {path}")
