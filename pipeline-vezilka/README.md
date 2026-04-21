# Pipeline Data Shapes And Final Ingestion

This document explains what data looks like at each phase of the `pipeline-vezilka` pipeline and what is finally ingested into LightRAG.

## Overview

Pipeline phases:

1. Phase 1: Discovery (`wikidata_discovery.py`)
2. Phase 2: Detail expansion (`wikidata_detail.py`)
3. Phase 3: Wikipedia enrichment (`wikipedia_fetch.py`)
4. Phase 4: Normalization (`normalization.py`)
5. Phase 5: Optional summarization (`summarization.py`)
6. Phase 6: Ingestion (`ingestion.py`)

Persisted artifacts:

- `data/discovered_qids.json`
- `data/entity_details.json`
- `data/normalized_documents.json`

## Phase 1 Output: Discovered QIDs

Produced by: `wikidata_discovery.py`
Saved to: `data/discovered_qids.json`

Shape:

```json
["Q185232", "Q50515696"]
```

Meaning:

- List of Wikidata IDs only.
- No labels/descriptions/properties yet.

## Phase 2 Output: Entity Details

Produced by: `wikidata_detail.py`
Saved to: `data/entity_details.json`

Shape (per entry):

```json
{
	"Q185232": {
		"qid": "Q185232",
		"label_mk": "...",
		"label_en": "...",
		"description_mk": "...",
		"description_en": "...",
		"sitelinks": {
			"mkwiki": "...",
			"enwiki": "..."
		},
		"properties": {
			"instance_of": ["..."],
			"occupation": ["..."],
			"date_of_birth": ["..."]
		}
	}
}
```

Meaning:

- Dictionary keyed by QID.
- Structured Wikidata fields + normalized property arrays.

## Phase 3 Output: Wikipedia-Enriched Entities

Produced by: `wikipedia_fetch.py`
Saved back to: `data/entity_details.json`

Additional fields added per entity:

- `wikipedia_extract`
- `wikipedia_full_text`
- `wikipedia_chunks`
- `wikipedia_chunk_size`
- `wikipedia_chunk_overlap`
- `wikipedia_source`
- `wikipedia_title`
- `wikipedia_url`
- `wikipedia_lang`
- `wikipedia_pageid`
- `wikipedia_selected_sections`
- `wikipedia_selection_strategy`

Meaning:

- Phase 2 record plus curated article content/chunks.

## Phase 4 Output: Normalized Documents (Main Ingestion Input)

Produced by: `normalization.py`
Saved to: `data/normalized_documents.json`

Shape:

```json
[
	{
		"qid": "Q185232",
		"text": "Deterministic full document text...",
		"metadata": {
			"qid": "Q185232",
			"label_en": "...",
			"label_mk": "...",
			"aliases": ["...", "..."],
			"canonical_entity_name": "...",
			"has_wikipedia": true,
			"wikipedia_url": "...",
			"text_lang": "en",
			"source": "wikidata+wikipedia"
		},
		"ingest_chunks": [
			"heading chunk",
			"wiki chunk 1",
			"wiki chunk 2",
			"facts chunk"
		]
	}
]
```

Meaning:

- `text`: deterministic human-readable full body.
- `ingest_chunks`: canonical chunk payload for ingestion.
- `metadata`: retrieval/filter/source metadata.

Additional metadata fields for bilingual pipelines:

- `aliases`: known cross-language names (mk/en) for the same entity.
- `canonical_entity_name`: default canonical graph name (mk when available).

## Phase 5 Output: Optional Summaries

Produced by: `summarization.py`
Saved back to: `data/normalized_documents.json`

Adds this optional key per document:

```json
"llm_summary": "3-5 sentence summary..."
```

Meaning:

- Optional compressed text for speed/cost fallback.
- Not recommended as the default ingestion source when retrieval quality is the goal.

## Phase 6: What Is Finally Ingested

Produced by: `ingestion.py`

For each normalized document:

- `id` = `qid`
- `file_path` = `metadata.wikipedia_url` (fallback: `wikidata://{qid}`)
- inserted text:
  - `llm_summary` when `prefer_llm_summary=true` and summary exists
  - otherwise joined `ingest_chunks`

Chunk join separator used for structured ingestion:

```text
\n\n<|LIGHTRAG_PIPELINE_CHUNK|>\n\n
```

Then ingestion calls `rag.ainsert(...)` with:

- `texts`
- `ids`
- `file_paths`

After insert, metadata is merged into `doc_status` rows via `rag.doc_status.upsert(...)`.

Cross-language deduplication:

- After insert, ingestion can merge mk/en duplicate entity nodes using `rag.amerge_entities(...)`.
- This is enabled by default via `LIGHTRAG_MERGE_CROSS_LANGUAGE_ALIASES=true`.
- Canonical target language is controlled by `LIGHTRAG_CANONICAL_ENTITY_LANG` (`mk` default, or `en`).

## Practical Summary

The final payload going into LightRAG is not raw Wikidata JSON.

It is:

- normalized deterministic text/chunks from Phase 4 (or optional Phase 5 summary),
- stable document IDs (`qid`),
- source file paths (Wikipedia URL fallback),
- and merged metadata for tracking/retrieval context.
