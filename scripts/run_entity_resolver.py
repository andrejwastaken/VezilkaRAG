#!/usr/bin/env python3
"""Run post-ingestion entity resolution for duplicate cleanup.

This script executes LightRAG's post-ingestion resolver and prints a JSON report.
It can be used manually or scheduled periodically (cron/systemd timer).
"""

from __future__ import annotations

import argparse
import asyncio
import json

from lightrag import LightRAG


async def _run(args: argparse.Namespace) -> None:
    rag = LightRAG()
    await rag.initialize_storages()
    try:
        result = await rag.aresolve_entities_post_ingestion(
            auto_merge_threshold=args.auto_merge_threshold,
            review_threshold=args.review_threshold,
            top_k=args.top_k,
            max_candidates=args.max_candidates,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        await rag.finalize_storages()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run post-ingestion entity resolution cleanup."
    )
    parser.add_argument(
        "--auto-merge-threshold",
        type=float,
        default=None,
        help="Override auto-merge threshold (default: value from .env / config).",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=None,
        help="Override review threshold (default: value from .env / config).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k vector neighbors per entity for candidate scoring.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=300,
        help="Maximum candidate pairs to score in one run.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
