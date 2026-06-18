#!/usr/bin/env python3
"""
Evaluate LightRAG/GraphRAG answers against a QA dataset.

The script runs each question through one or more LightRAG query modes, stores
the generated answers and retrieved contexts, and computes RAGAS plus lightweight
retrieval/generation/system metrics where the available data allows it.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import statistics
import sys
import time
import types
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience
    load_dotenv = None

try:
    import httpx
except ImportError:  # pragma: no cover - dependency check path
    httpx = None

if load_dotenv is not None:
    load_dotenv(dotenv_path=".env", override=False)


SYSTEM_MODE_MAP = {
    "naive": "naive",
    "local": "graph",
    "global": "graph",
    "hybrid": "mixed",
    "mix": "graph",
    "bypass": "bypass",
}

RAGAS_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "answer_similarity",
]

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "who",
    "where",
    "how",
    "и",
    "во",
    "на",
    "за",
    "со",
    "е",
    "се",
    "од",
    "што",
    "кој",
    "која",
}


@dataclass
class QACase:
    id: str
    question: str
    expected_answer: str
    source_ids: list[str] = field(default_factory=list)
    ground_truth_contexts: list[str] = field(default_factory=list)
    expected_entities: list[str] = field(default_factory=list)
    recommended_modes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u0400-\u04FF]+", clean_text(text).lower(), flags=re.UNICODE)


def content_words(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1]


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_source_ids(item: dict[str, Any]) -> list[str]:
    source_ids: list[str] = []
    for key in ("source_ids", "source_id", "reference_ids", "reference_id"):
        value = item.get(key)
        if isinstance(value, str):
            source_ids.extend([part.strip() for part in value.split(",") if part.strip()])
        elif isinstance(value, list):
            source_ids.extend(str(v).strip() for v in value if str(v).strip())

    for source in item.get("source_documents") or item.get("sources") or []:
        if isinstance(source, dict):
            for key in ("qid", "id", "source_id", "document_id", "doc_id", "file_path"):
                value = source.get(key)
                if value:
                    source_ids.append(str(value).strip())
                    break
        elif source:
            source_ids.append(str(source).strip())

    return sorted(set(source_ids))


def normalize_contexts(item: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    for key in (
        "ground_truth_contexts",
        "reference_contexts",
        "contexts",
        "evidence",
        "ground_truth_context",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            contexts.append(clean_text(value))
        elif isinstance(value, list):
            contexts.extend(clean_text(v) for v in value if clean_text(v))
    return contexts


def load_dataset(path: Path, limit: int | None = None) -> list[QACase]:
    raw = load_json(path)
    if isinstance(raw, dict):
        items = (
            raw.get("test_cases")
            or raw.get("data")
            or raw.get("items")
            or raw.get("questions")
            or []
        )
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported dataset root type: {type(raw).__name__}")

    cases: list[QACase] = []
    for idx, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        question = (
            item.get("question")
            or item.get("prasanje")
            or item.get("query")
            or item.get("input")
        )
        expected = (
            item.get("expected_answer")
            or item.get("right_answer")
            or item.get("ground_truth_answer")
            or item.get("ground_truth")
            or item.get("reference")
            or item.get("answer")
            or item.get("tocen_odgovor")
            or item.get("tochen_odgovor")
        )
        if not question or not expected:
            continue
        cases.append(
            QACase(
                id=str(item.get("id") or f"QA-{idx:04d}"),
                question=clean_text(question),
                expected_answer=clean_text(expected),
                source_ids=normalize_source_ids(item),
                ground_truth_contexts=normalize_contexts(item),
                expected_entities=[
                    clean_text(v)
                    for v in item.get("expected_entities", [])
                    if clean_text(v)
                ],
                recommended_modes=list(item.get("recommended_lightrag_modes") or []),
                metadata={
                    k: v
                    for k, v in item.items()
                    if k
                    not in {
                        "question",
                        "prasanje",
                        "query",
                        "input",
                        "expected_answer",
                        "right_answer",
                        "ground_truth_answer",
                        "ground_truth",
                        "reference",
                        "answer",
                        "tocen_odgovor",
                        "tochen_odgovor",
                    }
                },
            )
        )

    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError(f"No usable QA cases found in {path}")
    return cases


def exact_match(expected: str, generated: str) -> float:
    return float(clean_text(expected).casefold() == clean_text(generated).casefold())


def token_f1(expected: str, generated: str) -> float:
    expected_counts = Counter(tokenize(expected))
    generated_counts = Counter(tokenize(generated))
    if not expected_counts or not generated_counts:
        return 0.0
    overlap = sum((expected_counts & generated_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(generated_counts.values())
    recall = overlap / sum(expected_counts.values())
    return 2 * precision * recall / (precision + recall)


def lexical_semantic_similarity(expected: str, generated: str) -> float:
    expected_terms = set(content_words(expected))
    generated_terms = set(content_words(generated))
    if not expected_terms or not generated_terms:
        return 0.0
    return len(expected_terms & generated_terms) / len(expected_terms | generated_terms)


def relevant_context_ids(case: QACase) -> set[str]:
    return {src for src in case.source_ids if src}


def context_identifier_matches(identifier: str, relevant_ids: set[str]) -> bool:
    if not identifier:
        return False
    return any(rel and rel in identifier for rel in relevant_ids)


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def compute_retrieval_metrics(case: QACase, retrieved_ids: list[str], k: int) -> dict[str, Any]:
    relevant_ids = relevant_context_ids(case)
    top_ids = retrieved_ids[:k]
    if not relevant_ids:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "ndcg_at_k": None,
            "retrieval_metrics_note": "Skipped: dataset case has no ground-truth relevant document ids.",
        }

    hits = [1 if context_identifier_matches(identifier, relevant_ids) else 0 for identifier in top_ids]
    hit_count = sum(hits)
    first_hit = next((idx + 1 for idx, hit in enumerate(hits) if hit), None)
    ideal_hits = [1] * min(len(relevant_ids), k)
    ideal_dcg = dcg(ideal_hits)
    return {
        "precision_at_k": hit_count / k if k else None,
        "recall_at_k": min(hit_count, len(relevant_ids)) / len(relevant_ids),
        "mrr": 1 / first_hit if first_hit else 0.0,
        "ndcg_at_k": dcg(hits) / ideal_dcg if ideal_dcg else None,
        "retrieval_metrics_note": "",
    }


def estimate_tokens(text: str) -> int:
    # Cheap approximation used only when provider token usage is not available.
    return len(tokenize(text))


def extract_contexts_from_references(references: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    contexts: list[str] = []
    ids: list[str] = []
    for ref in references:
        ref_id = clean_text(ref.get("reference_id"))
        file_path = clean_text(ref.get("file_path"))
        identifier = " | ".join(part for part in (ref_id, file_path) if part)
        if identifier:
            ids.append(identifier)
        content = ref.get("content") or []
        if isinstance(content, str):
            content = [content]
        for chunk in content:
            if clean_text(chunk):
                contexts.append(clean_text(chunk))
    return contexts, ids


def extract_contexts_from_data(data: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    contexts: list[str] = []
    ids: list[str] = []
    graph = {
        "graph_nodes_retrieved": 0,
        "graph_edges_traversed": 0,
        "graph_average_depth_hops": None,
        "graph_coverage": None,
        "entity_recall": None,
        "graph_metrics_note": "",
    }

    for chunk in data.get("chunks") or []:
        if clean_text(chunk.get("content")):
            contexts.append(clean_text(chunk.get("content")))
        identifier = " | ".join(
            clean_text(chunk.get(key))
            for key in ("chunk_id", "reference_id", "file_path")
            if clean_text(chunk.get(key))
        )
        if identifier:
            ids.append(identifier)

    entities = data.get("entities") or []
    relationships = data.get("relationships") or []
    graph["graph_nodes_retrieved"] = len(entities)
    graph["graph_edges_traversed"] = len(relationships)

    for entity in entities:
        text = " ".join(
            clean_text(entity.get(key))
            for key in ("entity_name", "entity_type", "description", "file_path")
            if clean_text(entity.get(key))
        )
        if text:
            contexts.append(text)
        identifier = " | ".join(
            clean_text(entity.get(key))
            for key in ("entity_name", "source_id", "reference_id", "file_path")
            if clean_text(entity.get(key))
        )
        if identifier:
            ids.append(identifier)

    for rel in relationships:
        text = " ".join(
            clean_text(rel.get(key))
            for key in ("src_id", "tgt_id", "description", "keywords", "file_path")
            if clean_text(rel.get(key))
        )
        if text:
            contexts.append(text)
        identifier = " | ".join(
            clean_text(rel.get(key))
            for key in ("src_id", "tgt_id", "source_id", "reference_id", "file_path")
            if clean_text(rel.get(key))
        )
        if identifier:
            ids.append(identifier)

    for ref in data.get("references") or []:
        identifier = " | ".join(
            clean_text(ref.get(key))
            for key in ("reference_id", "file_path")
            if clean_text(ref.get(key))
        )
        if identifier:
            ids.append(identifier)

    graph["graph_metrics_note"] = (
        "LightRAG /query/data exposes entity and relationship counts, but not "
        "traversal depth/hops or complete graph coverage. To compute depth, "
        "return traversal path metadata. To compute graph/entity recall, add "
        "ground-truth expected graph nodes/entities to the QA dataset."
    )
    return contexts, ids, graph


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


async def post_json(
    client: Any,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    response = await client.post(url, json=payload, headers=headers or None)
    response.raise_for_status()
    return response.json()


async def run_case_mode(
    case: QACase,
    mode: str,
    endpoint: str,
    api_key: str | None,
    k: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    if httpx is None:
        raise ImportError("httpx is required to query the LightRAG API. Install project API/evaluation dependencies.")

    headers = {"X-API-Key": api_key} if api_key else {}
    payload_base = {
        "query": case.question,
        "mode": mode,
        "top_k": k,
        "chunk_top_k": k,
        "response_type": "Multiple Paragraphs",
        "include_references": True,
        "include_chunk_content": True,
    }
    started = time.perf_counter()
    result: dict[str, Any] = {
        "case_id": case.id,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "right_answer": case.expected_answer,
        "mode": mode,
        "system_type": SYSTEM_MODE_MAP.get(mode, mode),
        "generated_answer": "",
        "retrieved_contexts": [],
        "retrieved_context_ids": [],
        "latency_seconds": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "error": "",
    }

    timeout = httpx.Timeout(timeout_seconds, connect=min(30.0, timeout_seconds))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            answer_payload = dict(payload_base)
            answer_response = await post_json(
                client, f"{endpoint.rstrip('/')}/query", answer_payload, headers
            )
            result["generated_answer"] = clean_text(answer_response.get("response"))
            reference_contexts, reference_ids = extract_contexts_from_references(
                answer_response.get("references") or []
            )

            data_response = await post_json(
                client,
                f"{endpoint.rstrip('/')}/query/data",
                {"query": case.question, "mode": mode, "top_k": k, "chunk_top_k": k},
                headers,
            )
            data_contexts: list[str] = []
            data_ids: list[str] = []
            graph_metrics = {
                "graph_nodes_retrieved": 0,
                "graph_edges_traversed": 0,
                "graph_average_depth_hops": None,
                "graph_coverage": None,
                "entity_recall": None,
                "graph_metrics_note": "",
            }
            if data_response.get("status") == "success":
                data_contexts, data_ids, graph_metrics = extract_contexts_from_data(
                    data_response.get("data") or {}
                )
                processing_info = (data_response.get("metadata") or {}).get(
                    "processing_info"
                ) or {}
                result["lightrag_processing_info"] = processing_info
            else:
                result["error"] = clean_text(data_response.get("message"))

            result["retrieved_contexts"] = dedupe_preserve_order(
                reference_contexts + data_contexts
            )
            result["retrieved_context_ids"] = dedupe_preserve_order(reference_ids + data_ids)
            result.update(graph_metrics)

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["latency_seconds"] = round(time.perf_counter() - started, 4)

    result["total_tokens"] = estimate_tokens(case.question) + estimate_tokens(
        result["generated_answer"]
    )
    return result


def add_local_metrics(row: dict[str, Any], case: QACase, k: int) -> None:
    generated = row.get("generated_answer") or ""
    expected = case.expected_answer
    contexts = row.get("retrieved_contexts") or []
    retrieved_ids = row.get("retrieved_context_ids") or []

    row["exact_match"] = exact_match(expected, generated)
    row["token_f1"] = token_f1(expected, generated)
    row["semantic_similarity"] = lexical_semantic_similarity(expected, generated)
    row["judge_correctness"] = None
    row["judge_correctness_note"] = (
        "Skipped: no LLM judge implementation configured. RAGAS answer_correctness "
        "is used when RAGAS dependencies and evaluator LLM are available."
    )
    row["retrieved_chunks_count"] = len(contexts)
    row["empty_retrieval"] = float(len(contexts) == 0)
    row.update(compute_retrieval_metrics(case, retrieved_ids, k))
    if case.expected_entities:
        retrieved_text = " ".join(contexts + retrieved_ids).casefold()
        entity_hits = sum(1 for entity in case.expected_entities if entity.casefold() in retrieved_text)
        row["entity_recall"] = entity_hits / len(case.expected_entities)
    elif row.get("entity_recall") is None:
        row["entity_recall"] = None


def ragas_available() -> bool:
    ensure_ragas_import_compat()
    try:
        import datasets  # noqa: F401
        import ragas  # noqa: F401
        import ragas.metrics  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_ragas_import_compat() -> None:
    """Provide compatibility shims for optional provider paths imported by RAGAS."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        return
    shim = types.ModuleType(module_name)
    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


def run_ragas(rows: list[dict[str, Any]], output_dir: Path) -> str:
    if not rows:
        return "Skipped: no rows to evaluate."
    if not ragas_available():
        return (
            "Skipped: ragas/datasets are not installed. Install with "
            "`pip install -e .[evaluation]` or `pip install ragas datasets langchain-openai`."
        )
    if not (
        os.getenv("EVAL_LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
    ):
        return (
            "Skipped: EVAL_LLM_BINDING_API_KEY, OPENAI_API_KEY, or "
            "LLM_BINDING_API_KEY is not set."
        )

    ensure_ragas_import_compat()
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            AnswerSimilarity,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        return f"Skipped: missing RAGAS evaluation dependency: {exc}."

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
        AnswerSimilarity(),
    ]

    eval_llm_api_key = (
        os.getenv("EVAL_LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
    )
    eval_embedding_api_key = (
        os.getenv("EVAL_EMBEDDING_BINDING_API_KEY")
        or os.getenv("EVAL_LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("EMBEDDING_BINDING_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
    )
    eval_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
    eval_embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-small")
    llm_kwargs: dict[str, Any] = {
        "model": eval_model,
        "api_key": eval_llm_api_key,
        "max_retries": int(os.getenv("EVAL_LLM_MAX_RETRIES", "5")),
        "request_timeout": int(os.getenv("EVAL_LLM_TIMEOUT", "180")),
    }
    embedding_kwargs: dict[str, Any] = {
        "model": eval_embedding_model,
        "api_key": eval_embedding_api_key,
    }
    if os.getenv("EVAL_LLM_BINDING_HOST"):
        llm_kwargs["base_url"] = os.getenv("EVAL_LLM_BINDING_HOST")
    embedding_base_url = os.getenv("EVAL_EMBEDDING_BINDING_HOST") or os.getenv(
        "EVAL_LLM_BINDING_HOST"
    )
    if embedding_base_url:
        embedding_kwargs["base_url"] = embedding_base_url

    base_llm = ChatOpenAI(**llm_kwargs)
    try:
        eval_llm = LangchainLLMWrapper(langchain_llm=base_llm, bypass_n=True)
    except TypeError:
        eval_llm = LangchainLLMWrapper(langchain_llm=base_llm)
    eval_embeddings = OpenAIEmbeddings(**embedding_kwargs)

    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in rows],
            "answer": [row.get("generated_answer") or "" for row in rows],
            "contexts": [row.get("retrieved_contexts") or [] for row in rows],
            "ground_truth": [row.get("expected_answer") or "" for row in rows],
        }
    )
    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )
    df = eval_result.to_pandas()
    ragas_raw_path = output_dir / "ragas_raw_scores.csv"
    df.to_csv(ragas_raw_path, index=False)

    for idx, row in enumerate(rows):
        scores = df.iloc[idx].to_dict()
        for metric_name in RAGAS_METRIC_NAMES:
            row[metric_name] = safe_float(scores.get(metric_name))
        row["ragas_score"] = average_numbers(
            [row.get(metric_name) for metric_name in RAGAS_METRIC_NAMES]
        )
    return f"Computed RAGAS metrics; raw scores written to {ragas_raw_path}."


def average_numbers(values: list[Any]) -> float | None:
    nums = [safe_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    return sum(nums) / len(nums) if nums else None


def median_numbers(values: list[Any]) -> float | None:
    nums = [safe_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    return statistics.median(nums) if nums else None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            for key in ("retrieved_contexts", "retrieved_context_ids"):
                serializable[key] = json.dumps(serializable.get(key, []), ensure_ascii=False)
            writer.writerow(serializable)


def summarize(rows: list[dict[str, Any]], total_eval_time: float) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["system_type"]].append(row)

    metric_names = [
        *RAGAS_METRIC_NAMES,
        "ragas_score",
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "retrieved_chunks_count",
        "empty_retrieval",
        "exact_match",
        "token_f1",
        "semantic_similarity",
        "judge_correctness",
        "graph_nodes_retrieved",
        "graph_edges_traversed",
        "graph_average_depth_hops",
        "graph_coverage",
        "entity_recall",
        "latency_seconds",
        "total_tokens",
    ]

    summaries: list[dict[str, Any]] = []
    for system_type, system_rows in sorted(groups.items()):
        summary = {
            "system_type": system_type,
            "row_count": len(system_rows),
            "failed_query_count": sum(1 for row in system_rows if row.get("error")),
            "average_latency_seconds": average_numbers(
                [row.get("latency_seconds") for row in system_rows]
            ),
            "median_latency_seconds": median_numbers(
                [row.get("latency_seconds") for row in system_rows]
            ),
            "total_evaluation_time_seconds": round(total_eval_time, 4),
        }
        for metric_name in metric_names:
            summary[f"avg_{metric_name}"] = average_numbers(
                [row.get(metric_name) for row in system_rows]
            )
        summaries.append(summary)
    return summaries


def make_wide(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_question: dict[str, dict[str, Any]] = {}
    for row in rows:
        base = by_question.setdefault(
            row["question"],
            {
                "case_id": row.get("case_id"),
                "question": row["question"],
                "expected_answer": row["expected_answer"],
            },
        )
        prefix = row["system_type"]
        base[f"{prefix}_answer"] = row.get("generated_answer", "")
        base[f"{prefix}_faithfulness"] = row.get("faithfulness")
        base[f"{prefix}_answer_relevancy"] = row.get("answer_relevancy")
        base[f"{prefix}_context_precision"] = row.get("context_precision")
        base[f"{prefix}_context_recall"] = row.get("context_recall")
        base[f"{prefix}_answer_correctness"] = row.get("answer_correctness")
        base[f"{prefix}_latency"] = row.get("latency_seconds")
        base[f"{prefix}_token_f1"] = row.get("token_f1")
    return list(by_question.values())


def score_for_examples(row: dict[str, Any]) -> float:
    return average_numbers(
        [
            row.get("answer_correctness"),
            row.get("faithfulness"),
            row.get("token_f1"),
            row.get("semantic_similarity"),
        ]
    ) or 0.0


def create_report(
    path: Path,
    dataset_path: Path,
    cases: list[QACase],
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    modes: list[str],
    ragas_note: str,
    output_paths: dict[str, Path],
) -> None:
    systems = sorted({row["system_type"] for row in rows})
    summary_by_system = {row["system_type"]: row for row in summaries}
    comparison_metrics = [
        ("Faithfulness", "avg_faithfulness"),
        ("Answer Relevancy", "avg_answer_relevancy"),
        ("Context Precision", "avg_context_precision"),
        ("Context Recall", "avg_context_recall"),
        ("Answer Correctness", "avg_answer_correctness"),
        ("Token F1", "avg_token_f1"),
        ("Semantic Similarity", "avg_semantic_similarity"),
        ("Precision@K", "avg_precision_at_k"),
        ("Recall@K", "avg_recall_at_k"),
        ("MRR", "avg_mrr"),
        ("NDCG@K", "avg_ndcg_at_k"),
        ("Latency seconds", "average_latency_seconds"),
        ("Failed queries", "failed_query_count"),
    ]

    lines = [
        "# RAG Evaluation Report",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- QA cases: {len(cases)}",
        f"- Evaluated rows: {len(rows)}",
        f"- Evaluated modes: {', '.join(modes)}",
        f"- Evaluated systems: {', '.join(systems)}",
        "",
        "## Metrics",
        "",
        "- RAGAS: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness, answer_similarity when RAGAS and evaluator LLM credentials are available.",
        "- Retrieval: Precision@K, Recall@K, MRR, NDCG@K when ground-truth source ids exist; average retrieved chunks and empty retrieval rate always run.",
        "- Generation: exact match, token-level F1, lexical semantic similarity fallback. LLM judge correctness is TODO unless an evaluator judge is configured.",
        "- System: average latency, median latency, total time, failures, token estimate.",
        "- Graph: exposed entity/relationship counts are recorded. Depth/hops, graph coverage, and full entity recall require traversal metadata and ground-truth graph entities.",
        "",
        f"RAGAS status: {ragas_note}",
        "",
        "## Comparison",
        "",
        "| Metric | " + " | ".join(systems) + " |",
        "|---|" + "|".join("---:" for _ in systems) + "|",
    ]
    for label, key in comparison_metrics:
        values = []
        for system in systems:
            value = summary_by_system.get(system, {}).get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("N/A")
            else:
                values.append(str(value))
        lines.append("| " + label + " | " + " | ".join(values) + " |")

    failed = [row for row in rows if row.get("error")]
    best = sorted([row for row in rows if not row.get("error")], key=score_for_examples, reverse=True)[:3]
    worst = sorted([row for row in rows if not row.get("error")], key=score_for_examples)[:3]

    lines.extend(["", "## Best Examples", ""])
    lines.extend(format_examples(best))
    lines.extend(["", "## Worst Examples", ""])
    lines.extend(format_examples(worst))

    lines.extend(["", "## Failed Examples", ""])
    if failed:
        for row in failed[:10]:
            lines.append(f"- `{row.get('case_id')}` {row.get('system_type')}: {row.get('error')}")
    else:
        lines.append("- No failed examples.")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- RAGAS metrics are skipped if `ragas`, `datasets`, `langchain-openai`, or evaluator API keys are missing.",
            "- Precision@K/Recall@K/MRR/NDCG@K require source ids in the QA data and comparable ids in retrieved contexts.",
            "- Token usage is estimated unless the LightRAG API returns provider usage metadata.",
            "- Judge-based correctness is not computed unless an LLM judge is explicitly added.",
            "- Graph depth/hops and graph coverage need LightRAG to return traversal paths and ground-truth graph nodes/entities in the dataset.",
            "",
            "## Output Files",
            "",
        ]
    )
    for label, out_path in output_paths.items():
        lines.append(f"- {label}: `{out_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_examples(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- No examples available."]
    lines: list[str] = []
    for row in rows:
        contexts = row.get("retrieved_contexts") or []
        context_summary = clean_text(contexts[0])[:240] if contexts else "No retrieved contexts."
        lines.extend(
            [
                f"### {row.get('case_id')} / {row.get('system_type')}",
                "",
                f"- Question: {row.get('question')}",
                f"- Expected answer: {row.get('expected_answer')}",
                f"- Generated answer: {row.get('generated_answer')}",
                f"- Retrieved contexts summary: {context_summary}",
                f"- Score signal: token_f1={format_optional(row.get('token_f1'))}, faithfulness={format_optional(row.get('faithfulness'))}, answer_correctness={format_optional(row.get('answer_correctness'))}",
                "- Explanation: Higher scores indicate closer answer overlap and, when available, stronger RAGAS grounding/correctness. Review the retrieved context to decide whether retrieval or generation caused the gap.",
                "",
            ]
        )
    return lines


def format_optional(value: Any) -> str:
    number = safe_float(value)
    return "N/A" if number is None else f"{number:.4f}"


LONG_FIELDS = [
    "case_id",
    "question",
    "expected_answer",
    "right_answer",
    "generated_answer",
    "retrieved_contexts",
    "retrieved_context_ids",
    "system_type",
    "mode",
    "latency_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "answer_similarity",
    "ragas_score",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "retrieved_chunks_count",
    "empty_retrieval",
    "exact_match",
    "token_f1",
    "semantic_similarity",
    "judge_correctness",
    "graph_nodes_retrieved",
    "graph_edges_traversed",
    "graph_average_depth_hops",
    "graph_coverage",
    "entity_recall",
    "error",
]


def build_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "per_question_results.csv": output_dir / "per_question_results.csv",
        "per_question_results.json": output_dir / "per_question_results.json",
        "summary_metrics.csv": output_dir / "summary_metrics.csv",
        "summary_metrics.json": output_dir / "summary_metrics.json",
        "qa_comparison_wide.csv": output_dir / "qa_comparison_wide.csv",
        "evaluation_report.md": output_dir / "evaluation_report.md",
    }


def write_outputs(
    *,
    output_paths: dict[str, Path],
    dataset_path: Path,
    cases: list[QACase],
    rows: list[dict[str, Any]],
    modes: list[str],
    total_eval_time: float,
    ragas_note: str,
) -> None:
    summaries = summarize(rows, total_eval_time)
    wide_rows = make_wide(rows)
    summary_fields = sorted({key for row in summaries for key in row.keys()})
    wide_fields = sorted({key for row in wide_rows for key in row.keys()})

    write_csv(output_paths["per_question_results.csv"], rows, LONG_FIELDS)
    output_paths["per_question_results.json"].write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_paths["summary_metrics.csv"], summaries, summary_fields)
    output_paths["summary_metrics.json"].write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_paths["qa_comparison_wide.csv"], wide_rows, wide_fields)
    create_report(
        path=output_paths["evaluation_report.md"],
        dataset_path=dataset_path,
        cases=cases,
        rows=rows,
        summaries=summaries,
        modes=modes,
        ragas_note=ragas_note,
        output_paths=output_paths,
    )


def parse_modes(raw_modes: str, use_recommended: bool = False) -> list[str]:
    if use_recommended:
        return []
    modes = [mode.strip() for mode in raw_modes.split(",") if mode.strip()]
    allowed = set(SYSTEM_MODE_MAP)
    invalid = [mode for mode in modes if mode not in allowed]
    if invalid:
        raise ValueError(f"Unsupported mode(s): {', '.join(invalid)}")
    return modes


async def evaluate(args: argparse.Namespace) -> dict[str, Path]:
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_dataset(dataset_path, args.limit)
    static_modes = parse_modes(args.modes, args.use_recommended_modes)
    report_modes = static_modes or ["recommended-per-question"]
    api_key = args.api_key or os.getenv("LIGHTRAG_API_KEY")
    output_paths = build_output_paths(output_dir)

    all_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if args.from_results:
        results_path = Path(args.from_results)
        loaded_rows = load_json(results_path)
        if not isinstance(loaded_rows, list):
            raise ValueError(f"--from-results must point to a JSON list: {results_path}")
        all_rows = [row for row in loaded_rows if isinstance(row, dict)]
        print(f"Loaded {len(all_rows)} rows from {results_path}")
    else:
        for idx, case in enumerate(cases, 1):
            modes = static_modes or [
                mode for mode in case.recommended_modes if mode in SYSTEM_MODE_MAP
            ] or ["naive", "mix"]
            for mode in modes:
                print(f"[{idx}/{len(cases)}] {case.id} mode={mode}")
                row = await run_case_mode(
                    case=case,
                    mode=mode,
                    endpoint=args.endpoint,
                    api_key=api_key,
                    k=args.k,
                    timeout_seconds=args.timeout,
                )
                add_local_metrics(row, case, args.k)
                all_rows.append(row)
                if args.checkpoint_every and len(all_rows) % args.checkpoint_every == 0:
                    write_outputs(
                        output_paths=output_paths,
                        dataset_path=dataset_path,
                        cases=cases,
                        rows=all_rows,
                        modes=report_modes,
                        total_eval_time=time.perf_counter() - started,
                        ragas_note=(
                            "Pending: checkpoint written before RAGAS scoring. "
                            "Use --skip-ragas to avoid evaluator LLM calls."
                        ),
                    )
    total_eval_time = time.perf_counter() - started

    write_outputs(
        output_paths=output_paths,
        dataset_path=dataset_path,
        cases=cases,
        rows=all_rows,
        modes=report_modes,
        total_eval_time=total_eval_time,
        ragas_note=(
            "Pending: generation and local metrics completed; RAGAS has not run yet. "
            "This checkpoint is safe to use for non-RAGAS metrics."
        ),
    )

    ragas_note = "Skipped by --skip-ragas."
    if not args.skip_ragas:
        try:
            ragas_note = run_ragas(all_rows, output_dir)
        except Exception as exc:
            ragas_note = (
                f"Failed: {type(exc).__name__}: {exc}. "
                "Generation/local-metric outputs were preserved before RAGAS."
            )

    write_outputs(
        output_paths=output_paths,
        dataset_path=dataset_path,
        cases=cases,
        rows=all_rows,
        modes=report_modes,
        total_eval_time=total_eval_time,
        ragas_note=ragas_note,
    )
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LightRAG/GraphRAG with RAGAS and additional metrics.",
        epilog=(
            "Install notes: run `pip install -e .[evaluation]` for RAGAS, "
            "datasets, langchain-openai, and HTTP dependencies. Set "
            "LIGHTRAG_API_KEY for the LightRAG API if auth is enabled. Set "
            "EVAL_LLM_BINDING_API_KEY or OPENAI_API_KEY for RAGAS metrics. "
            "Use --skip-ragas for a retrieval/generation smoke test without "
            "evaluator LLM calls."
        ),
    )
    parser.add_argument("--dataset", required=True, help="Path to QA dataset JSON.")
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory where evaluation outputs will be written.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit.")
    parser.add_argument("--k", type=int, default=5, help="Top K for retrieval metrics.")
    parser.add_argument(
        "--modes",
        default="naive,mix,hybrid",
        help="Comma-separated LightRAG modes to evaluate.",
    )
    parser.add_argument(
        "--use-recommended-modes",
        action="store_true",
        help="Use each QA item's recommended_lightrag_modes instead of --modes.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("LIGHTRAG_API_URL", "http://localhost:9621"),
        help="LightRAG API endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LightRAG API key. Defaults to LIGHTRAG_API_KEY from environment.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds per API request.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS scoring and only compute local metrics.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help=(
            "Write intermediate CSV/JSON/report outputs every N evaluated rows. "
            "Use 0 to disable checkpoints."
        ),
    )
    parser.add_argument(
        "--from-results",
        default=None,
        help=(
            "Load an existing per_question_results.json and compute outputs/RAGAS "
            "without querying the LightRAG API again."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started_at = datetime.now().isoformat()
    print(f"Starting RAG evaluation at {started_at}")
    try:
        output_paths = asyncio.run(evaluate(args))
    except Exception as exc:
        raise SystemExit(f"Evaluation failed: {type(exc).__name__}: {exc}") from exc
    print("Generated files:")
    for path in output_paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
