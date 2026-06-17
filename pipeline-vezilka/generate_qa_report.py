"""
Generate a grounded QA evaluation set for the Vezilka LightRAG corpus.

The generator is intentionally deterministic and uses only collected pipeline
artifacts, so every answer remains traceable to normalized documents/chunks.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "normalized_documents.json"
OUT_JSON = REPO_ROOT / "data" / "vezilka_qa_eval_100.json"
OUT_CSV = REPO_ROOT / "data" / "vezilka_qa_eval_100.csv"
OUT_REPORT = REPO_ROOT / "docs" / "vezilka_qa_eval_report.md"

MODES_ALL = ["naive", "local", "global", "hybrid", "mix"]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def label(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    return (
        metadata.get("label_en")
        or metadata.get("label_mk")
        or metadata.get("canonical_entity_name")
        or doc["qid"]
    )


def mk_label(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    return metadata.get("label_mk") or label(doc)


def source_ref(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = doc.get("metadata") or {}
    return {
        "qid": doc["qid"],
        "label_en": metadata.get("label_en"),
        "label_mk": metadata.get("label_mk"),
        "wikipedia_url": metadata.get("wikipedia_url"),
    }


def sentence_split(text: str) -> list[str]:
    text = clean(text)
    protected = {
        "с.": "с<dot>",
        "р.": "р<dot>",
        "г.": "г<dot>",
        "ул.": "ул<dot>",
        "бр.": "бр<dot>",
    }
    for needle, replacement in protected.items():
        text = text.replace(needle, replacement)
    pieces = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for piece in pieces:
        for needle, replacement in protected.items():
            piece = piece.replace(replacement, needle)
        piece = piece.strip()
        if len(piece) > 20:
            sentences.append(piece)
    return sentences


def chunk_sentences(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(doc.get("ingest_chunks") or [], 1):
        for sent in sentence_split(chunk):
            rows.append({"chunk": idx, "sentence": sent})
    return rows


def first_sentence(doc: dict[str, Any]) -> str:
    chunks = doc.get("ingest_chunks") or []
    if chunks:
        return sentence_split(chunks[0])[0]
    return sentence_split(doc.get("text") or "")[0]


PATTERNS = {
    "birth_or_life_dates": re.compile(
        r"\b(was born|born \d|born on|роден е|родена е|е роден|е родена|роден во|родена во|роден на|родена на)\b",
        re.I,
    ),
    "location": re.compile(r"\b(located|се наоѓа|се наоѓаат|сместен|сместена|во градот|near|близина)\b", re.I),
    "construction_or_foundation": re.compile(
        r"\b(built|constructed|founded|изградена|изграден|подигната|подигнат|основан|основана)\b",
        re.I,
    ),
    "publication": re.compile(r"\b(published|објавен|објавена|издаден|издадена|книга|роман|збирка)\b", re.I),
    "career_or_award": re.compile(r"\b(champion|winner|career|награда|добитник|кариера|медал|професор)\b", re.I),
}


def best_sentence(doc: dict[str, Any], kind: str) -> dict[str, Any] | None:
    pattern = PATTERNS[kind]
    for row in chunk_sentences(doc):
        if pattern.search(row["sentence"]):
            return row
    return None


def add_item(
    items: list[dict[str, Any]],
    seen_questions: set[str],
    *,
    question: str,
    answer: str,
    query_type: str,
    answer_type: str,
    sources: list[dict[str, Any]],
    evidence: list[str],
    modes: list[str],
) -> None:
    question = clean(question)
    answer = clean(answer)
    if not question or not answer or question in seen_questions:
        return
    seen_questions.add(question)
    items.append(
        {
            "id": f"VZ-QA-{len(items) + 1:03d}",
            "question": question,
            "answer": answer,
            "query_type": query_type,
            "answer_type": answer_type,
            "recommended_lightrag_modes": modes,
            "source_documents": sources,
            "evidence": [clean(e) for e in evidence],
        }
    )


def generate_items(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = sorted(docs, key=lambda d: (0 if (d.get("metadata") or {}).get("has_wikipedia") else 1, d["qid"]))
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        for instance in (doc.get("metadata") or {}).get("instance_of") or []:
            by_instance[instance].append(doc)

    items: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    # Single-hop / single-chunk identity questions.
    for doc in docs:
        metadata = doc.get("metadata") or {}
        desc = metadata.get("description")
        instances = ", ".join(metadata.get("instance_of") or [])
        answer = desc or first_sentence(doc)
        if instances and desc:
            answer = f"{label(doc)} is described as {desc}; its instance type is {instances}."
        add_item(
            items,
            seen_questions,
            question=f"What is {label(doc)} in the Vezilka corpus?",
            answer=answer,
            query_type="single-hop / single-chunk factual",
            answer_type="short entity description",
            sources=[source_ref(doc)],
            evidence=[first_sentence(doc)],
            modes=["naive", "local", "hybrid", "mix"],
        )
        if len(items) >= 18:
            break

    # Bilingual alias questions.
    for doc in docs:
        if len(items) >= 30:
            break
        metadata = doc.get("metadata") or {}
        en = metadata.get("label_en")
        mk = metadata.get("label_mk")
        if en and mk and en.casefold() != mk.casefold():
            add_item(
                items,
                seen_questions,
                question=f"Which English/Latin entity label corresponds to the Macedonian label {mk}?",
                answer=f"{mk} corresponds to {en}.",
                query_type="single-hop bilingual alias",
                answer_type="alias mapping",
                sources=[source_ref(doc)],
                evidence=[f"aliases: {', '.join(metadata.get('aliases') or [])}"],
                modes=["local", "hybrid", "mix"],
            )

    # Patterned fact questions grounded in exact sentences.
    templates = {
        "birth_or_life_dates": "What biographical date or life-detail is stated for {name}?",
        "location": "Where is {name} located or situated according to the corpus?",
        "construction_or_foundation": "What construction or foundation detail is stated for {name}?",
        "publication": "What publication or book detail is stated for {name}?",
        "career_or_award": "What career, award, or achievement detail is stated for {name}?",
    }
    answer_types = {
        "birth_or_life_dates": "date/place biographical fact",
        "location": "location fact",
        "construction_or_foundation": "historical date/event fact",
        "publication": "bibliographic fact",
        "career_or_award": "career or award fact",
    }
    targeted_goal_by_kind = {
        "birth_or_life_dates": 7,
        "location": 7,
        "construction_or_foundation": 7,
        "publication": 7,
        "career_or_award": 7,
    }
    for kind, question_template in templates.items():
        added_for_kind = 0
        for doc in docs:
            if added_for_kind >= targeted_goal_by_kind[kind]:
                break
            row = best_sentence(doc, kind)
            if not row:
                continue
            before = len(items)
            add_item(
                items,
                seen_questions,
                question=question_template.format(name=label(doc)),
                answer=row["sentence"],
                query_type="single-hop / targeted factual",
                answer_type=answer_types[kind],
                sources=[source_ref(doc)],
                evidence=[row["sentence"]],
                modes=["naive", "local", "hybrid", "mix"],
            )
            if len(items) > before:
                added_for_kind += 1

    # Multi-chunk questions inside one document.
    for doc in docs:
        if len(items) >= 80:
            break
        intro = first_sentence(doc)
        secondary = None
        for kind in ("location", "construction_or_foundation", "career_or_award", "publication"):
            row = best_sentence(doc, kind)
            if row and row["sentence"] != intro:
                secondary = row
                break
        if not secondary:
            continue
        add_item(
            items,
            seen_questions,
            question=f"Summarize {label(doc)} using both its corpus description and one detailed supporting fact.",
            answer=f"{intro} Supporting detail: {secondary['sentence']}",
            query_type="single-hop / multi-chunk synthesis",
            answer_type="two-part synthesized answer",
            sources=[source_ref(doc)],
            evidence=[intro, secondary["sentence"]],
            modes=["hybrid", "mix", "local"],
        )

    # Cross-document comparisons by shared instance type.
    pair_candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for instance, group in sorted(by_instance.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        group = [g for g in group if (g.get("metadata") or {}).get("has_wikipedia")]
        for i in range(0, len(group) - 1, 2):
            pair_candidates.append((instance, group[i], group[i + 1]))
    for instance, left, right in pair_candidates:
        if len(items) >= 92:
            break
        left_fact = first_sentence(left)
        right_fact = first_sentence(right)
        add_item(
            items,
            seen_questions,
            question=f"Compare {label(left)} and {label(right)} as entities of type {instance}. What is each one?",
            answer=f"{label(left)}: {left_fact} {label(right)}: {right_fact}",
            query_type="multi-hop / cross-document comparison",
            answer_type="comparative two-entity answer",
            sources=[source_ref(left), source_ref(right)],
            evidence=[left_fact, right_fact],
            modes=["global", "hybrid", "mix"],
        )

    # Multi-hop relation-style questions with two different evidence clauses.
    used_set_questions: set[str] = set()
    for instance, group in sorted(by_instance.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(group) < 3:
            continue
        for group_start in range(0, len(group) - 2, 3):
            if len(items) >= 100:
                break
            trio = group[group_start : group_start + 3]
            trio_key = ",".join(d["qid"] for d in trio)
            if trio_key in used_set_questions:
                continue
            used_set_questions.add(trio_key)
            answer = "; ".join(f"{label(d)} is represented as {first_sentence(d)}" for d in trio)
            add_item(
                items,
                seen_questions,
                question=f"Name three Vezilka corpus entities with instance type {instance} and state the core fact for each.",
                answer=answer,
                query_type="multi-hop / set retrieval",
                answer_type="enumerated multi-entity answer",
                sources=[source_ref(d) for d in trio],
                evidence=[first_sentence(d) for d in trio],
                modes=["global", "hybrid", "mix"],
            )
        if len(items) >= 100:
            break

    if len(items) < 100:
        raise RuntimeError(f"Generated only {len(items)} QA items")
    return items[:100]


def write_outputs(items: list[dict[str, Any]], docs: list[dict[str, Any]]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    OUT_JSON.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "answer",
                "query_type",
                "answer_type",
                "recommended_lightrag_modes",
                "source_qids",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "query_type": item["query_type"],
                    "answer_type": item["answer_type"],
                    "recommended_lightrag_modes": ", ".join(item["recommended_lightrag_modes"]),
                    "source_qids": ", ".join(src["qid"] for src in item["source_documents"]),
                }
            )

    query_counts = Counter(item["query_type"] for item in items)
    answer_counts = Counter(item["answer_type"] for item in items)
    mode_counts = Counter(mode for item in items for mode in item["recommended_lightrag_modes"])
    source_qids = {src["qid"] for item in items for src in item["source_documents"]}
    has_wikipedia = sum(1 for d in docs if (d.get("metadata") or {}).get("has_wikipedia"))
    total_chunks = sum(len(d.get("ingest_chunks") or []) for d in docs)

    lines = [
        "# Vezilka LightRAG QA Evaluation Report",
        "",
        "## Corpus Basis",
        "",
        f"- Source artifact: `{DATA_FILE.relative_to(REPO_ROOT)}`",
        "- Target LightRAG workspace observed in Postgres: `vezilka_limit6`",
        f"- Normalized documents: {len(docs)}",
        f"- Documents with Wikipedia enrichment: {has_wikipedia}",
        f"- Ingest chunks in normalized artifact: {total_chunks}",
        f"- QA items generated: {len(items)}",
        f"- Unique source QIDs used by QA items: {len(source_qids)}",
        "",
        "## Intended Retrieval Coverage",
        "",
        "- `naive`: direct chunk/vector facts and simple factual questions.",
        "- `local`: entity-neighborhood and alias-sensitive questions.",
        "- `global`: cross-document comparisons and set retrieval.",
        "- `hybrid`: combined local/global questions.",
        "- `mix`: graph plus vector retrieval, especially multi-chunk and multi-hop items.",
        "- `bypass`: not included as a primary target because it bypasses retrieval; the dataset is designed to evaluate retrieval-grounded answers.",
        "",
        "## Query Type Summary",
        "",
        "| Query type | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {kind} | {count} |" for kind, count in sorted(query_counts.items()))
    lines.extend(
        [
            "",
            "## Answer Type Summary",
            "",
            "| Answer type | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {kind} | {count} |" for kind, count in sorted(answer_counts.items()))
    lines.extend(
        [
            "",
            "## Mode Coverage",
            "",
            "| Mode | QA items marked suitable |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {mode} | {mode_counts[mode]} |" for mode in ["naive", "local", "global", "hybrid", "mix"])
    lines.extend(
        [
            "",
            "## Full QA Set",
            "",
            "| ID | Question | Answer | Query type | Answer type | Modes | Sources |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in items:
        sources = ", ".join(src["qid"] for src in item["source_documents"])
        modes = ", ".join(item["recommended_lightrag_modes"])
        row = [
            item["id"],
            item["question"],
            item["answer"],
            item["query_type"],
            item["answer_type"],
            modes,
            sources,
        ]
        escaped = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")

    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    docs = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    items = generate_items(docs)
    write_outputs(items, docs)
    print(f"Wrote {len(items)} QA items")
    print(OUT_JSON)
    print(OUT_CSV)
    print(OUT_REPORT)


if __name__ == "__main__":
    main()
