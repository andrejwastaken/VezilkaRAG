"""
Phase 4 – Normalization Layer
==============================

Converts structured entity JSON (produced by Phase 2 + 3) into:
  - A deterministic, factual text paragraph (RAG-ready, no LLM required)
  - A clean metadata dict for filtered retrieval in LightRAG

Design principles:
  - Pure functions; no side effects
  - Deterministic output for the same input
    - Wikipedia chunked text is preferred; summary is fallback
  - Structured facts follow in a consistent order
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _join(vals: List[str], limit: Optional[int] = None) -> str:
    """Join a list of values into a readable comma/and string."""
    if limit:
        vals = vals[:limit]
    if len(vals) == 1:
        return vals[0]
    return ", ".join(vals[:-1]) + " and " + vals[-1]


# ---------------------------------------------------------------------------
# Sentence renderers
#
# Each renderer is  fn(entity_name: str, values: List[str]) -> str
# Two dicts – one per language.  Properties are rendered in declaration order.
# ---------------------------------------------------------------------------

RENDERERS_EN: Dict[str, Any] = {
    "instance_of":             lambda n, v: f"{n} is a {_join(v, 3)}.",
    "subclass_of":             lambda n, v: f"{n} is a subclass of {_join(v, 3)}.",
    "part_of":                 lambda n, v: f"{n} is part of {_join(v, 3)}.",
    "has_part":                lambda n, v: f"{n} includes {_join(v, 3)}.",
    "date_of_birth":           lambda n, v: f"{n} was born on {v[0]}.",
    "date_of_death":           lambda n, v: f"{n} died on {v[0]}.",
    "start_time":              lambda n, v: f"{n} started on {v[0]}.",
    "end_time":                lambda n, v: f"{n} ended on {v[0]}.",
    "place_of_birth":          lambda n, v: f"{n} was born in {_join(v, 2)}.",
    "place_of_death":          lambda n, v: f"{n} died in {_join(v, 2)}.",
    "country_of_citizenship":  lambda n, v: f"{n} is a citizen of {_join(v, 3)}.",
    "country_of_origin":       lambda n, v: f"{n} originates from {_join(v, 2)}.",
    "country":                 lambda n, v: f"{n} is associated with {_join(v, 2)}.",
    "occupation":              lambda n, v: f"{n}'s occupation: {_join(v, 5)}.",
    "author":                  lambda n, v: f"Authored by {_join(v, 3)}.",
    "genre":                   lambda n, v: f"{n} works in the genre(s): {_join(v, 5)}.",
    "movement":                lambda n, v: f"{n} is part of the {_join(v, 3)} movement.",
    "award_received":          lambda n, v: f"{n} has received: {_join(v, 8)}.",
    "official_language":       lambda n, v: f"Official language: {_join(v, 3)}.",
    "notable_work":            lambda n, v: f"Notable works: {_join(v, 5)}.",
    "record_label":            lambda n, v: f"{n} is signed to {_join(v, 3)}.",
    "director":                lambda n, v: f"Directed by {_join(v, 3)}.",
    "cast_member":             lambda n, v: f"Features {_join(v, 5)} in the cast.",
    "screenwriter":            lambda n, v: f"Written by {_join(v, 3)}.",
    "original_language":       lambda n, v: f"Original language: {_join(v)}.",
    "language_of_work":        lambda n, v: f"Language: {_join(v)}.",
    "writing_system":          lambda n, v: f"Writing system: {_join(v, 3)}.",
    "iso_639_1_code":          lambda n, v: f"ISO 639-1 code: {_join(v, 3)}.",
    "iso_639_2_code":          lambda n, v: f"ISO 639-2 code: {_join(v, 3)}.",
    "iso_639_3_code":          lambda n, v: f"ISO 639-3 code: {_join(v, 3)}.",
    "inception":               lambda n, v: f"{n} was established on {v[0]}.",
    "founded_by":              lambda n, v: f"Founded by {_join(v, 3)}.",
    "located_in":              lambda n, v: f"{n} is located in {_join(v, 2)}.",
    "location":                lambda n, v: f"Location: {_join(v, 2)}.",
    "headquarters_location":   lambda n, v: f"Headquarters: {_join(v, 2)}.",
    "place_of_publication":    lambda n, v: f"Place of publication: {_join(v, 3)}.",
    "coordinate_location":     lambda n, v: f"Coordinates: {v[0]}.",
    "official_website":        lambda n, v: f"Official website: {_join(v, 2)}.",
    "member_of":               lambda n, v: f"{n} is a member of {_join(v, 3)}.",
    "educated_at":             lambda n, v: f"{n} was educated at {_join(v, 3)}.",
    "employer":                lambda n, v: f"{n} worked for {_join(v, 3)}.",
    "influenced_by":           lambda n, v: f"{n} was influenced by {_join(v, 5)}.",
    "sport":                   lambda n, v: f"{n} is associated with {_join(v, 2)}.",
    "member_of_sports_team":   lambda n, v: f"{n} played for {_join(v, 3)}.",
    "position_played":         lambda n, v: f"{n} played as {_join(v, 2)}.",
    "performer":               lambda n, v: f"Performed by {_join(v, 3)}.",
    "composer":                lambda n, v: f"Music composed by {_join(v, 3)}.",
    "creator":                 lambda n, v: f"Created by {_join(v, 3)}.",
    "publisher":               lambda n, v: f"Published by {_join(v, 3)}.",
    "publication_date":        lambda n, v: f"{n} was published on {v[0]}.",
    "main_subject":            lambda n, v: f"Main subject: {_join(v, 5)}.",
}

RENDERERS_MK: Dict[str, Any] = {
    "instance_of":             lambda n, v: f"{n} е {_join(v, 3)}.",
    "subclass_of":             lambda n, v: f"{n} е подкласа на {_join(v, 3)}.",
    "part_of":                 lambda n, v: f"{n} е дел од {_join(v, 3)}.",
    "has_part":                lambda n, v: f"{n} вклучува {_join(v, 3)}.",
    "date_of_birth":           lambda n, v: f"{n} е роден/а на {v[0]}.",
    "date_of_death":           lambda n, v: f"{n} почина на {v[0]}.",
    "start_time":              lambda n, v: f"{n} започнал/а на {v[0]}.",
    "end_time":                lambda n, v: f"{n} завршил/а на {v[0]}.",
    "place_of_birth":          lambda n, v: f"{n} е роден/а во {_join(v, 2)}.",
    "place_of_death":          lambda n, v: f"{n} почина во {_join(v, 2)}.",
    "country_of_citizenship":  lambda n, v: f"{n} е државјанин/ка на {_join(v, 3)}.",
    "country_of_origin":       lambda n, v: f"{n} потекнува од {_join(v, 2)}.",
    "country":                 lambda n, v: f"{n} е поврзан/а со {_join(v, 2)}.",
    "occupation":              lambda n, v: f"Занимање на {n}: {_join(v, 5)}.",
    "author":                  lambda n, v: f"Автор: {_join(v, 3)}.",
    "genre":                   lambda n, v: f"{n} работи во жанр(ови): {_join(v, 5)}.",
    "movement":                lambda n, v: f"{n} припаѓа на движењето {_join(v, 3)}.",
    "award_received":          lambda n, v: f"{n} добил/а: {_join(v, 8)}.",
    "official_language":       lambda n, v: f"Официјален јазик: {_join(v, 3)}.",
    "notable_work":            lambda n, v: f"Значајни дела: {_join(v, 5)}.",
    "record_label":            lambda n, v: f"{n} е потпишан/а за {_join(v, 3)}.",
    "director":                lambda n, v: f"Режирано од {_join(v, 3)}.",
    "cast_member":             lambda n, v: f"Во улоги: {_join(v, 5)}.",
    "screenwriter":            lambda n, v: f"Напишано од {_join(v, 3)}.",
    "original_language":       lambda n, v: f"Оригинален јазик: {_join(v)}.",
    "language_of_work":        lambda n, v: f"Јазик: {_join(v)}.",
    "writing_system":          lambda n, v: f"Писмо: {_join(v, 3)}.",
    "iso_639_1_code":          lambda n, v: f"ISO 639-1 код: {_join(v, 3)}.",
    "iso_639_2_code":          lambda n, v: f"ISO 639-2 код: {_join(v, 3)}.",
    "iso_639_3_code":          lambda n, v: f"ISO 639-3 код: {_join(v, 3)}.",
    "inception":               lambda n, v: f"{n} е основан/а на {v[0]}.",
    "founded_by":              lambda n, v: f"Основано од {_join(v, 3)}.",
    "located_in":              lambda n, v: f"{n} се наоѓа во {_join(v, 2)}.",
    "location":                lambda n, v: f"Локација: {_join(v, 2)}.",
    "headquarters_location":   lambda n, v: f"Седиште: {_join(v, 2)}.",
    "place_of_publication":    lambda n, v: f"Место на издавање: {_join(v, 3)}.",
    "coordinate_location":     lambda n, v: f"Координати: {v[0]}.",
    "official_website":        lambda n, v: f"Официјална веб-страница: {_join(v, 2)}.",
    "member_of":               lambda n, v: f"{n} е член на {_join(v, 3)}.",
    "educated_at":             lambda n, v: f"{n} се школувал/а на {_join(v, 3)}.",
    "employer":                lambda n, v: f"{n} работел/а за {_join(v, 3)}.",
    "influenced_by":           lambda n, v: f"{n} бил/а под влијание на {_join(v, 5)}.",
    "sport":                   lambda n, v: f"{n} е поврзан/а со {_join(v, 2)}.",
    "member_of_sports_team":   lambda n, v: f"{n} играл/а за {_join(v, 3)}.",
    "position_played":         lambda n, v: f"{n} играл/а на позиција {_join(v, 2)}.",
    "performer":               lambda n, v: f"Изведено од {_join(v, 3)}.",
    "composer":                lambda n, v: f"Музика компонирана од {_join(v, 3)}.",
    "creator":                 lambda n, v: f"Создадено од {_join(v, 3)}.",
    "publisher":               lambda n, v: f"Издадено од {_join(v, 3)}.",
    "publication_date":        lambda n, v: f"{n} е издаден/а на {v[0]}.",
    "main_subject":            lambda n, v: f"Главна тема: {_join(v, 5)}.",
}

# Keep old name as alias so any external code referencing RENDERERS still works
RENDERERS = RENDERERS_EN

LANG_RENDERERS: Dict[str, Dict[str, Any]] = {"en": RENDERERS_EN, "mk": RENDERERS_MK}


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def build_document(entity: Dict[str, Any], lang: str = "en") -> Tuple[str, Dict[str, Any]]:
    """
    Convert one entity dict into (text_paragraph, metadata).

    Args:
        entity: Structured entity dict from Phase 2/3.
        lang:   ``"en"`` (default) or ``"mk"`` – controls which label,
                description, and sentence templates are used.
                Wikipedia text is inserted verbatim regardless of this setting
                (already fetched in the best available language).

    text_paragraph structure:
        1. Heading sentence  (name + description)
        2. Wikipedia chunked text when available; otherwise summary extract
        3. Structured fact sentences (one per property, in ``lang``)

    metadata is language-independent (always contains both label_mk and label_en).
    """
    if lang not in LANG_RENDERERS:
        raise ValueError(f"lang must be 'en' or 'mk', got {lang!r}")
    renderers = LANG_RENDERERS[lang]

    qid: str = entity["qid"]

    # Name used inside sentences: prefer the requested language
    if lang == "mk":
        name: str = entity.get("label_mk") or entity.get("label_en") or qid
        desc: str = entity.get("description_mk") or entity.get("description_en") or ""
        fallback_sentence = f"{name} е културен субјект од Северна Македонија."
    else:
        name = entity.get("label_en") or entity.get("label_mk") or qid
        desc = entity.get("description_en") or entity.get("description_mk") or ""
        fallback_sentence = f"{name} is a cultural entity from North Macedonia."

    props: Dict[str, List[str]] = entity.get("properties", {})
    wiki_chunks: List[str] = entity.get("wikipedia_chunks") or []
    wiki_text: str = "\n\n".join(wiki_chunks) if wiki_chunks else (
        entity.get("wikipedia_full_text") or entity.get("wikipedia_extract", "")
    )

    lines: List[str] = []

    # ── 1. Heading ────────────────────────────────────────────────────────────
    if desc:
        lines.append(f"{name} — {desc}.")
    else:
        lines.append(fallback_sentence)

    # ── 2. Wikipedia text (full preferred, summary fallback) ────────────────
    if wiki_text:
        lines.append("")
        lines.append(wiki_text)

    # ── 3. Structured facts ───────────────────────────────────────────────────
    lines.append("")
    for prop_key, renderer in renderers.items():
        vals = props.get(prop_key)
        if not vals:
            continue
        sentence = renderer(name, vals)
        lines.append(sentence)

    # Strip trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()

    text: str = "\n".join(lines).strip()

    # ── Metadata ──────────────────────────────────────────────────────────────
    instance_types = props.get("instance_of", [])
    country = (
        props.get("country_of_citizenship")
        or props.get("country_of_origin")
        or props.get("country")
        or []
    )

    metadata: Dict[str, Any] = {
        "qid":             qid,
        "label_mk":        entity.get("label_mk"),
        "label_en":        entity.get("label_en"),
        "description":     entity.get("description_en") or entity.get("description_mk"),
        "instance_of":     instance_types[:3],
        "country":         country[:1],
        "has_wikipedia":   bool(wiki_text),
        "wikipedia_source": entity.get("wikipedia_source"),
        "wikipedia_title": entity.get("wikipedia_title"),
        "wikipedia_pageid": entity.get("wikipedia_pageid"),
        "wikipedia_url":   entity.get("wikipedia_url"),
        "wikipedia_lang":  entity.get("wikipedia_lang"),
        "wikipedia_chunk_count": len(wiki_chunks),
        "wikipedia_chunk_size": entity.get("wikipedia_chunk_size"),
        "wikipedia_chunk_overlap": entity.get("wikipedia_chunk_overlap"),
        "wikipedia_text_chars": len(wiki_text),
        "text_lang":       lang,
        "source":          "wikidata+wikipedia",
    }

    return text, metadata


def normalize_all(
    entities: Dict[str, Any],
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """
    Normalize all entities.

    Args:
        entities: {qid: entity_dict} from Phase 2/3.
        lang:     ``"en"`` or ``"mk"`` – language for sentence templates and
                  which label/description to prefer in the text body.

    Returns:
        List of {"qid", "text", "metadata"} dicts.
        Entities that produce empty documents are silently skipped.
    """
    if lang not in ("en", "mk"):
        raise ValueError(f"lang must be 'en' or 'mk', got {lang!r}")
    documents: List[Dict[str, Any]] = []
    for qid, entity in entities.items():
        try:
            text, metadata = build_document(entity, lang=lang)
            if text.strip():
                documents.append({"qid": qid, "text": text, "metadata": metadata})
        except Exception as exc:
            print(f"  Normalization error for {qid}: {exc}")
    return documents


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    import argparse
    parser = argparse.ArgumentParser(description="Normalize entity details into RAG documents.")
    parser.add_argument("--lang", choices=["en", "mk"], default="en",
                        help="Language for sentence templates (default: en).")
    args = parser.parse_args()

    src = DATA_DIR / "entity_details.json"
    if not src.exists():
        print(f"Run wikidata_detail.py (and optionally wikipedia_fetch.py) first to generate {src}.")
    else:
        entities = json.loads(src.read_text(encoding="utf-8"))
        docs = normalize_all(entities, lang=args.lang)
        out = DATA_DIR / "normalized_documents.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Normalized {len(docs)} documents (lang={args.lang}) → {out}")

        # Preview first document
        if docs:
            print("\n── Sample document ──────────────────────────────────────")
            print(docs[0]["text"][:800])
            print("────────────────────────────────────────────────────────")
