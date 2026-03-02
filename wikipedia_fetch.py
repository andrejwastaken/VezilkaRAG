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

import time
import urllib.parse
from typing import Dict, Optional

import requests

MK_WIKI_SUMMARY = "https://mk.wikipedia.org/api/rest_v1/page/summary/{title}"
EN_WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

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


# ---------------------------------------------------------------------------
# Enrichment loop
# ---------------------------------------------------------------------------

def enrich_with_wikipedia(
    entities: Dict[str, dict],
    delay: float = REQUEST_DELAY,
) -> Dict[str, dict]:
    """
    Adds a ``wikipedia_extract`` key to each entity that has a Wikipedia
    article.  Macedonian is tried first; English is used as a fallback.

    Mutates *entities* in-place and returns it.
    """
    total = len(entities)
    found = 0

    for idx, (qid, ent) in enumerate(entities.items(), 1):
        sitelinks = ent.get("sitelinks", {})
        mk_title: Optional[str] = sitelinks.get("mkwiki")
        en_title: Optional[str] = sitelinks.get("enwiki")

        extract: Optional[str] = None
        source: Optional[str] = None

        if mk_title:
            extract = fetch_summary(mk_title, lang="mk")
            if extract:
                source = "mk.wikipedia"

        if not extract and en_title:
            extract = fetch_summary(en_title, lang="en")
            if extract:
                source = "en.wikipedia"

        if extract:
            ent["wikipedia_extract"] = extract
            ent["wikipedia_source"] = source
            found += 1
            label = ent.get("label_en") or ent.get("label_mk") or qid
            print(
                f"  [{idx:>4}/{total}] {qid}  {label[:40]:<40}  "
                f"{source}  ({len(extract)} chars)"
            )
        else:
            pass  # silent skip – keeps output manageable for large runs

        time.sleep(delay)

    print(f"\nWikipedia extracts found: {found}/{total}")
    return entities


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import pathlib

    path = pathlib.Path("data/entity_details.json")
    if not path.exists():
        print("Run wikidata_detail.py first to generate data/entity_details.json.")
    else:
        entities = json.loads(path.read_text(encoding="utf-8"))
        entities = enrich_with_wikipedia(entities)
        path.write_text(json.dumps(entities, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Updated {len(entities)} entities → {path}")
