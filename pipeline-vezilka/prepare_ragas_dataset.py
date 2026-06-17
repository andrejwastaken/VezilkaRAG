"""
Convert the Vezilka QA evaluation set to the dataset shape used by
``lightrag/evaluation/eval_rag_quality.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "vezilka_qa_eval_100.json"
DEFAULT_OUTPUT = REPO_ROOT / "lightrag" / "evaluation" / "vezilka_qa_ragas_dataset.json"


def convert_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "question": item["question"],
        "ground_truth": item["answer"],
        "project": "vezilka_lightrag",
        "query_type": item.get("query_type"),
        "answer_type": item.get("answer_type"),
        "recommended_lightrag_modes": item.get("recommended_lightrag_modes", []),
        "source_documents": item.get("source_documents", []),
        "evidence": item.get("evidence", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Vezilka QA data for the LightRAG RAGAS evaluator."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of QA cases to export for smoke tests.",
    )
    args = parser.parse_args()

    items = json.loads(args.input.read_text(encoding="utf-8"))
    if args.limit is not None:
        items = items[: args.limit]

    dataset = {"test_cases": [convert_item(item) for item in items]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(dataset['test_cases'])} test cases to {args.output}")


if __name__ == "__main__":
    main()
