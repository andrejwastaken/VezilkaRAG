"""
Phase 2 – Wikidata Detail Expansion
=====================================

Fetches structured entity data (labels, descriptions, claims) from the
Wikidata REST API for each QID.  Returns raw structured JSON – no text
rendering is done here.

API docs: https://www.wikidata.org/w/api.php
Batch size: up to 50 IDs per request (Wikidata hard limit).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
BATCH_SIZE = 50
REQUEST_DELAY = 1.0

HEADERS = {"User-Agent": "MacedonianCulturePipeline/1.0 (research@example.com)"}

# ---------------------------------------------------------------------------
# Property map  (Wikidata PID → friendly key used throughout the pipeline)
# ---------------------------------------------------------------------------
PROPERTY_MAP: Dict[str, str] = {
    "P31":   "instance_of",
    "P21":   "sex_or_gender",
    "P27":   "country_of_citizenship",
    "P106":  "occupation",
    "P136":  "genre",
    "P569":  "date_of_birth",
    "P570":  "date_of_death",
    "P19":   "place_of_birth",
    "P20":   "place_of_death",
    "P166":  "award_received",
    "P135":  "movement",
    "P364":  "original_language",
    "P495":  "country_of_origin",
    "P17":   "country",
    "P571":  "inception",
    "P131":  "located_in",
    "P276":  "location",
    "P264":  "record_label",
    "P57":   "director",
    "P161":  "cast_member",
    "P58":   "screenwriter",
    "P407":  "language_of_work",
    "P108":  "employer",
    "P69":   "educated_at",
    "P800":  "notable_work",
    "P463":  "member_of",
    "P737":  "influenced_by",
    "P413":  "position_played",
    "P54":   "member_of_sports_team",
    "P641":  "sport",
    "P607":  "conflict",
    "P585":  "point_in_time",
    "P175":  "performer",
    "P86":   "composer",
    "P170":  "creator",
    "P710":  "participant",
    "P123":  "publisher",
    "P577":  "publication_date",
}


# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------

def _api_get(params: dict) -> dict:
    resp = requests.get(
        WIKIDATA_API,
        params={**params, "format": "json"},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _get_label(entity_data: dict, lang: str = "mk") -> Optional[str]:
    labels = entity_data.get("labels", {})
    if lang in labels:
        return labels[lang]["value"]
    if "en" in labels:
        return labels["en"]["value"]
    return None


def _get_description(entity_data: dict, lang: str = "mk") -> Optional[str]:
    descs = entity_data.get("descriptions", {})
    if lang in descs:
        return descs[lang]["value"]
    if "en" in descs:
        return descs["en"]["value"]
    return None


def _resolve_value(snak: dict, labels_cache: dict) -> Optional[str]:
    """Extract a human-readable string from a single Wikidata snak."""
    if snak.get("snaktype") != "value":
        return None

    dv = snak.get("datavalue", {})
    dt = dv.get("type")

    if dt == "wikibase-entityid":
        qid = dv["value"]["id"]
        return labels_cache.get(qid, qid)  # fall back to raw QID if label missing

    if dt == "string":
        return dv["value"]

    if dt == "time":
        # "+YYYY-MM-DDT00:00:00Z"  →  "YYYY-MM-DD"
        raw: str = dv["value"]["time"]
        return raw.lstrip("+").split("T")[0]

    if dt == "monolingualtext":
        return dv["value"]["text"]

    if dt == "quantity":
        return str(dv["value"]["amount"])

    if dt == "globecoordinate":
        lat = dv["value"]["latitude"]
        lon = dv["value"]["longitude"]
        return f"{lat},{lon}"

    return None


# ---------------------------------------------------------------------------
# Two-pass label resolution
# ---------------------------------------------------------------------------

def _collect_value_qids(entities_raw: dict) -> List[str]:
    """Collect QIDs that appear as claim values so we can batch-resolve their labels."""
    qids: set = set()
    for entity_data in entities_raw.values():
        if not isinstance(entity_data, dict):
            continue
        for pid in PROPERTY_MAP:
            for claim in entity_data.get("claims", {}).get(pid, []):
                snak = claim.get("mainsnak", {})
                if snak.get("snaktype") == "value":
                    dv = snak.get("datavalue", {})
                    if dv.get("type") == "wikibase-entityid":
                        qids.add(dv["value"]["id"])
    return list(qids)


def _batch_fetch_labels(qids: List[str]) -> Dict[str, str]:
    """Fetch mk/en labels for a list of QIDs in BATCH_SIZE chunks."""
    labels: Dict[str, str] = {}
    for i in range(0, len(qids), BATCH_SIZE):
        batch = qids[i : i + BATCH_SIZE]
        try:
            data = _api_get({
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels",
                "languages": "mk|en",
            })
            for qid, ent in data.get("entities", {}).items():
                lbl = (
                    ent.get("labels", {}).get("mk", {}).get("value")
                    or ent.get("labels", {}).get("en", {}).get("value")
                )
                if lbl:
                    labels[qid] = lbl
        except Exception:
            pass
        time.sleep(REQUEST_DELAY)
    return labels


# ---------------------------------------------------------------------------
# Core batch fetch
# ---------------------------------------------------------------------------

def fetch_entities_batch(qids: List[str]) -> Dict[str, Any]:
    """
    Fetch full entity data for up to BATCH_SIZE QIDs.
    Returns {qid: structured_dict}.
    """
    data = _api_get({
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "labels|descriptions|claims|sitelinks",
        "languages": "mk|en",
        "sitefilter": "mkwiki|enwiki",
    })
    entities_raw = data.get("entities", {})

    # Resolve value-QID labels in a second pass
    value_qids = _collect_value_qids(entities_raw)
    labels_cache = _batch_fetch_labels(value_qids) if value_qids else {}

    result: Dict[str, Any] = {}
    for qid, ent in entities_raw.items():
        if ent.get("missing") == "":
            continue

        structured: Dict[str, Any] = {
            "qid": qid,
            "label_mk": _get_label(ent, "mk"),
            "label_en": _get_label(ent, "en"),
            "description_mk": _get_description(ent, "mk"),
            "description_en": _get_description(ent, "en"),
            "sitelinks": {
                k: v["title"]
                for k, v in ent.get("sitelinks", {}).items()
                if k in ("mkwiki", "enwiki")
            },
            "properties": {},
        }

        for pid, friendly in PROPERTY_MAP.items():
            claims = ent.get("claims", {}).get(pid, [])
            values: List[str] = []
            for claim in claims:
                val = _resolve_value(claim.get("mainsnak", {}), labels_cache)
                if val and val not in values:
                    values.append(val)
            if values:
                structured["properties"][friendly] = values

        result[qid] = structured

    return result


def fetch_all_entities(
    qids: List[str],
    delay: float = REQUEST_DELAY,
) -> Dict[str, Any]:
    """
    Fetch details for all QIDs, batching BATCH_SIZE at a time.
    Returns {qid: structured_dict}.
    """
    all_entities: Dict[str, Any] = {}
    total = len(qids)

    for i in range(0, total, BATCH_SIZE):
        batch = qids[i : i + BATCH_SIZE]
        end = min(i + BATCH_SIZE, total)
        print(f"  Fetching {i + 1}–{end} / {total} ...", end=" ", flush=True)
        try:
            batch_result = fetch_entities_batch(batch)
            all_entities.update(batch_result)
            print(f"OK ({len(batch_result)} entities)")
        except Exception as exc:
            print(f"ERROR – {exc}")
        time.sleep(delay)

    return all_entities


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import pathlib

    qids_path = pathlib.Path("data/discovered_qids.json")
    if not qids_path.exists():
        print("Run wikidata_discovery.py first to generate data/discovered_qids.json.")
    else:
        qids = json.loads(qids_path.read_text(encoding="utf-8"))
        print(f"Fetching details for {len(qids)} QIDs …")
        result = fetch_all_entities(qids)
        out = pathlib.Path("data/entity_details.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(result)} entities → {out}")
