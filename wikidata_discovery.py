"""
Phase 1 – Wikidata Discovery
============================

Discovers Macedonian cultural entity QIDs via Wikidata SPARQL.
Uses P31/P279* (instance/subclass) expansion and country filters.

North Macedonia = Q221
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import requests

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
NORTH_MACEDONIA_QID = "Q221"
REQUEST_DELAY = 2.0
LIMIT_PER_CATEGORY = 500

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------
# Each entry is one of:
#   human=True  → match wd:Q5 (human) whose P106/P279* = occupation_qid
#                 filtered by country_prop = Q221
#   human=False → match P31/P279* type_qid filtered by country_prop = Q221
#
# country_prop:
#   wdt:P27  – country of citizenship          (people)
#   wdt:P495 – country of origin               (films, albums, groups)
#   wdt:P17  – country                         (places, organisations)
# ---------------------------------------------------------------------------

CATEGORIES: List[Dict[str, Any]] = [
    # ── PEOPLE ────────────────────────────────────────────────────────────────
    {"name": "Artists",        "human": True, "occupation_qid": "Q483501",  "country_prop": "wdt:P27"},
    {"name": "Musicians",      "human": True, "occupation_qid": "Q639669",  "country_prop": "wdt:P27"},
    {"name": "Singers",        "human": True, "occupation_qid": "Q177220",  "country_prop": "wdt:P27"},
    {"name": "Film directors", "human": True, "occupation_qid": "Q2526255", "country_prop": "wdt:P27"},
    {"name": "Actors",         "human": True, "occupation_qid": "Q33999",   "country_prop": "wdt:P27"},
    {"name": "Writers",        "human": True, "occupation_qid": "Q36180",   "country_prop": "wdt:P27"},
    {"name": "Painters",       "human": True, "occupation_qid": "Q1028181", "country_prop": "wdt:P27"},
    {"name": "Sculptors",      "human": True, "occupation_qid": "Q1281618", "country_prop": "wdt:P27"},
    {"name": "Architects",     "human": True, "occupation_qid": "Q42973",   "country_prop": "wdt:P27"},
    {"name": "Poets",          "human": True, "occupation_qid": "Q49757",   "country_prop": "wdt:P27"},
    {"name": "Composers",      "human": True, "occupation_qid": "Q36834",   "country_prop": "wdt:P27"},
    {"name": "Athletes",       "human": True, "occupation_qid": "Q2066131", "country_prop": "wdt:P27"},
    {"name": "Scientists",     "human": True, "occupation_qid": "Q901",     "country_prop": "wdt:P27"},
    {"name": "Politicians",    "human": True, "occupation_qid": "Q82955",   "country_prop": "wdt:P27"},
    {"name": "Journalists",    "human": True, "occupation_qid": "Q1930187", "country_prop": "wdt:P27"},
    {"name": "Photographers",  "human": True, "occupation_qid": "Q33231",   "country_prop": "wdt:P27"},
    # ── NON-PERSON ENTITIES ───────────────────────────────────────────────────
    {"name": "Films",          "human": False, "type_qid": "Q11424",    "country_prop": "wdt:P495"},
    {"name": "Music groups",   "human": False, "type_qid": "Q215380",   "country_prop": "wdt:P495"},
    {"name": "Albums",         "human": False, "type_qid": "Q482994",   "country_prop": "wdt:P495"},
    {"name": "Museums",        "human": False, "type_qid": "Q33506",    "country_prop": "wdt:P17"},
    {"name": "Theatres",       "human": False, "type_qid": "Q24354",    "country_prop": "wdt:P17"},
    {"name": "Festivals",      "human": False, "type_qid": "Q132241",   "country_prop": "wdt:P17"},
    {"name": "TV shows",       "human": False, "type_qid": "Q5398426",  "country_prop": "wdt:P495"},
    {"name": "Universities",   "human": False, "type_qid": "Q3918",     "country_prop": "wdt:P17"},
    {"name": "Sports teams",   "human": False, "type_qid": "Q12973014", "country_prop": "wdt:P17"},
    {"name": "Libraries",      "human": False, "type_qid": "Q7075",     "country_prop": "wdt:P17"},
    {"name": "Newspapers",     "human": False, "type_qid": "Q11032",    "country_prop": "wdt:P17"},
]


# ---------------------------------------------------------------------------
# SPARQL utilities
# ---------------------------------------------------------------------------

def sparql_query(query: str) -> dict:
    resp = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "MacedonianCulturePipeline/1.0 (research@example.com)",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _human_query(occupation_qid: str, country_prop: str, limit: int) -> str:
    return f"""
SELECT DISTINCT ?entity WHERE {{
  ?entity wdt:P31 wd:Q5 .
  ?entity wdt:P106/wdt:P279* wd:{occupation_qid} .
  ?entity {country_prop} wd:{NORTH_MACEDONIA_QID} .
}}
LIMIT {limit}
"""


def _type_query(type_qid: str, country_prop: str, limit: int) -> str:
    return f"""
SELECT DISTINCT ?entity WHERE {{
  ?entity wdt:P31/wdt:P279* wd:{type_qid} .
  ?entity {country_prop} wd:{NORTH_MACEDONIA_QID} .
}}
LIMIT {limit}
"""


def _extract_qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------

def discover_entities(limit_per_category: int = LIMIT_PER_CATEGORY) -> List[str]:
    """
    Run all category queries against Wikidata and return a sorted,
    deduplicated list of QIDs.
    """
    discovered: set = set()

    for cat in CATEGORIES:
        name = cat["name"]
        country_prop = cat["country_prop"]

        if cat.get("human"):
            query = _human_query(cat["occupation_qid"], country_prop, limit_per_category)
        else:
            query = _type_query(cat["type_qid"], country_prop, limit_per_category)

        print(f"  Discovering: {name:<22}", end=" ")
        try:
            data = sparql_query(query)
            qids = [
                _extract_qid(b["entity"]["value"])
                for b in data["results"]["bindings"]
            ]
            discovered.update(qids)
            print(f"{len(qids):>4} found  (total so far: {len(discovered)})")
        except Exception as exc:
            print(f"ERROR – {exc}")

        time.sleep(REQUEST_DELAY)

    return sorted(discovered)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import pathlib

    qids = discover_entities()
    out = pathlib.Path("data/discovered_qids.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(qids, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(qids)} QIDs → {out}")
