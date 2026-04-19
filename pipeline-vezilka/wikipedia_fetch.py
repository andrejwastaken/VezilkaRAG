"""
Phase 3 – Wikipedia Article Fetch
=================================

Fetches Wikipedia text for entities discovered in earlier phases, but does not
blindly ingest entire articles. The goal is queryable retrieval text, not raw
encyclopedia dumps.

Strategy:
  - fetch the lead summary via REST API
  - fetch plaintext article content via MediaWiki
  - keep the lead plus a small number of informative sections
  - skip boilerplate sections such as references and external links
  - chunk the curated text for downstream normalization/ingestion

This produces denser chunks, reduces ingestion cost, and avoids arbitrary
character truncation in the middle of an article.
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

DEFAULT_CHUNK_SIZE_TOKENS = 450
DEFAULT_CHUNK_OVERLAP_TOKENS = 60
DEFAULT_TARGET_TOKENS = 3200
DEFAULT_MAX_SECTION_TOKENS = 900
DEFAULT_MAX_SECTIONS = 4

CHUNK_SIZE_TOKENS = int(os.getenv("LIGHTRAG_WIKI_CHUNK_TOKENS", str(DEFAULT_CHUNK_SIZE_TOKENS)))
CHUNK_OVERLAP_TOKENS = int(
    os.getenv("LIGHTRAG_WIKI_CHUNK_OVERLAP_TOKENS", str(DEFAULT_CHUNK_OVERLAP_TOKENS))
)
TARGET_TEXT_TOKENS = int(os.getenv("LIGHTRAG_WIKI_TARGET_TOKENS", str(DEFAULT_TARGET_TOKENS)))
MAX_SECTION_TOKENS = int(
    os.getenv("LIGHTRAG_WIKI_MAX_SECTION_TOKENS", str(DEFAULT_MAX_SECTION_TOKENS))
)
MAX_SECTIONS = int(os.getenv("LIGHTRAG_WIKI_MAX_SECTIONS", str(DEFAULT_MAX_SECTIONS)))

SECTION_HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", flags=re.MULTILINE)
IGNORED_SECTION_TITLES = {
    "see also",
    "references",
    "notes",
    "citations",
    "sources",
    "further reading",
    "external links",
    "bibliography",
    "works cited",
    "publications",
    "selected works",
    "discography",
    "filmography",
    "gallery",
}

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
    """Fetch plaintext article content via MediaWiki action API."""
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
                "full_text": raw,
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


def _normalize_section_title(title: str) -> str:
    # Normalize section titles for stable ignore-list matching.
    return re.sub(r"\s+", " ", title).strip().lower()


def _clean_section_body(body: str) -> str:
    # Remove empty lines and trailing whitespace from section bodies.
    lines = [line.rstrip() for line in body.splitlines()]
    cleaned = "\n".join(line for line in lines if line.strip()).strip()
    return cleaned


def _truncate_to_tokens(text: str, token_limit: int) -> str:
    # Trim text to a token budget while preserving sentence boundaries.
    if token_limit <= 0 or _token_count(text) <= token_limit:
        return text

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text

    kept: list[str] = []
    total = 0
    for sentence in sentences:
        sentence_tokens = _token_count(sentence)
        if kept and total + sentence_tokens > token_limit:
            break
        kept.append(sentence)
        total += sentence_tokens
        if total >= token_limit:
            break
    return " ".join(kept).strip() or text


def _split_article_sections(text: str) -> list[dict[str, Any]]:
    """Split plaintext wiki extract into lead + heading-based sections."""
    cleaned = text.strip()
    if not cleaned:
        return []

    matches = list(SECTION_HEADING_RE.finditer(cleaned))
    if not matches:
        return [{"title": "Lead", "text": cleaned, "level": 0}]

    sections: list[dict[str, Any]] = []
    lead_text = cleaned[: matches[0].start()].strip()
    if lead_text:
        sections.append({"title": "Lead", "text": lead_text, "level": 0})

    for idx, match in enumerate(matches):
        heading = match.group(2).strip()
        level = len(match.group(1))
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        body = _clean_section_body(cleaned[body_start:body_end])
        if not body:
            continue
        sections.append({"title": heading, "text": body, "level": level})

    return sections


def _select_queryable_sections(
    full_text: str,
    summary: Optional[str],
) -> tuple[str, list[str]]:
    """
    Keep a bounded, informative article slice for retrieval.

    The selected text should answer common entity questions quickly:
    who/what it is, why it matters, key dates/places/works, and major context.
    """
    sections = _split_article_sections(full_text)
    if not sections:
        return summary or "", []

    chosen_blocks: list[str] = []
    chosen_titles: list[str] = []
    budget = max(TARGET_TEXT_TOKENS, CHUNK_SIZE_TOKENS)

    lead_section = sections[0]
    lead_text = _truncate_to_tokens(lead_section["text"], min(MAX_SECTION_TOKENS, budget))
    if lead_text:
        chosen_blocks.append(lead_text)
        chosen_titles.append(lead_section["title"])
        budget -= _token_count(lead_text)

    informative_count = 0
    for section in sections[1:]:
        if informative_count >= MAX_SECTIONS or budget <= CHUNK_OVERLAP_TOKENS:
            break

        normalized_title = _normalize_section_title(section["title"])
        if normalized_title in IGNORED_SECTION_TITLES:
            continue

        section_budget = min(MAX_SECTION_TOKENS, budget)
        section_text = _truncate_to_tokens(section["text"], section_budget)
        if not section_text or _token_count(section_text) < 40:
            continue

        chosen_blocks.append(f"{section['title']}\n{section_text}".strip())
        chosen_titles.append(section["title"])
        informative_count += 1
        budget -= _token_count(section_text)

    curated_text = "\n\n".join(block for block in chosen_blocks if block.strip()).strip()
    if not curated_text:
        return summary or "", []
    return curated_text, chosen_titles


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

    raw_full_text = (full or {}).get("full_text")
    full_text, selected_sections = (
        _select_queryable_sections(raw_full_text, summary) if raw_full_text else (summary or "", [])
    )
    chunks = _chunk_text_for_lightrag(full_text) if full_text else []

    return {
        "summary": summary,
        "full_text": full_text,
        "raw_full_text": raw_full_text,
        "chunks": chunks,
        "title": resolved_title,
        "pageid": (full or {}).get("pageid"),
        "url": page_template.format(title=encoded_title),
        "lang": lang,
        "selected_sections": selected_sections,
        "selection_strategy": "lead_plus_informative_sections",
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
            ent["wikipedia_selected_sections"] = article.get("selected_sections") or []
            ent["wikipedia_selection_strategy"] = article.get("selection_strategy")

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
