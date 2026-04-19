# OpenAI Pipeline Smoke Test Run

## Goal

Run the Macedonian cultural knowledge pipeline from phase 1 on the OpenAI-backed path with the minimum practical scope, confirm that each phase executes, and verify that ingested data can be queried successfully.

## Configuration Used

Runtime path:

- Docker Compose stack for API + PostgreSQL
- Ad hoc container run for the pipeline
- Fresh workspace: `openai_smoke_min1`

Model configuration:

- `LLM_BINDING=openai`
- `LLM_MODEL=gpt-4.1-mini`
- `EMBEDDING_BINDING=openai`
- `EMBEDDING_MODEL=text-embedding-3-small`

Minimal-scope choice:

- Pipeline launched with `--limit 1`

Important interpretation of `--limit 1`:

- This limits discovery to one result per category.
- It does **not** mean one total document.
- Because the discovery script iterates over many categories, the run still produced a diverse smoke-test corpus.

## Phase Choices

### Normalization

- Enabled

Reason:

- Normalization is required to convert the structured Wikidata/Wikipedia entity payloads into deterministic RAG-ready documents.

### Summarization

- Disabled

Reason:

- For the OpenAI path, the goal of this smoke test was to preserve retrieval quality rather than optimize ingestion speed.
- Earlier local evaluation showed that summary-backed ingestion is faster, but can weaken retrieval quality.

## Command Used

The pipeline was started from phase 1 with:

```bash
/app/.venv/bin/python -u pipeline-vezilka/pipeline.py --limit 1
```

The run used the fresh workspace `openai_smoke_min1`.

## Phase-By-Phase Results

### Phase 1. Wikidata Discovery

Result:

- Success
- `48` unique QIDs discovered

Notes:

- Some categories returned `0`, which is expected.
- With `--limit 1`, the run still covered people, places, institutions, media, monuments, and related entities.

### Phase 2. Detail Expansion

Result:

- Success
- `48` entities fetched successfully

Notes:

- Structured Wikidata labels, descriptions, sitelinks, and property claims were created for all discovered QIDs.

### Phase 3. Wikipedia Article Fetch

Result:

- Success
- `42/48` entities had Wikipedia article content

Notes:

- This phase showed why even the minimal-scope run is still non-trivial.
- Some pages were small.
- Some pages hit the article cap and produced many chunks.

Examples observed during the run:

- `Tetovo` -> `30000` chars, `55` chunks
- `Toše Proeski` -> `30000` chars, `59` chunks
- `Leb i sol` -> `30000` chars, `58` chunks

Interpretation:

- The OpenAI path handled these large pages much more cleanly than the local Ollama path.
- But they still make the run materially larger than a tiny one-document demo.

### Phase 4. Normalization

Result:

- Success
- `48` normalized documents produced

Notes:

- This phase was intentionally kept on.
- It produced the structured ingestion text used by phase 6.

### Phase 5. Summarization

Result:

- Intentionally skipped

Reason:

- This smoke test prioritized structured ingestion quality over reduced extraction cost.

### Phase 6. LightRAG Ingestion

Result:

- Success in the sense that ingestion started cleanly and processed documents correctly
- The run was intentionally stopped after end-to-end validation to keep OpenAI spend low

Observed storage backend:

- PostgreSQL-backed KV/doc status/graph/vector storage

Observed vector tables:

- `LIGHTRAG_VDB_ENTITY_text_embedding_3_small_1536d`
- `LIGHTRAG_VDB_RELATION_text_embedding_3_small_1536d`
- `LIGHTRAG_VDB_CHUNKS_text_embedding_3_small_1536d`

Observed behavior:

- No timeout failures were observed during the tested portion
- No ingestion crashes were observed
- Documents moved steadily into `processed`

Final workspace counts when the batch was stopped:

- `22 processed`
- `22 pending`
- `4 processing`

Reason the batch was stopped:

- The objective of this run was validation, not full-corpus completion
- After proving phase 1 through phase 6 worked and confirming successful retrieval, continuing would only spend more API quota

## Retrieval Validation

### API workspace behavior

An application behavior issue was observed:

- The `LIGHTRAG-WORKSPACE` header was honored by `/health`
- But the query path still behaved as though it was tied to the server default workspace rather than the requested workspace

Observed consequence:

- Querying the running API against `openai_smoke_min1` returned no context, even though the fresh workspace had already ingested relevant documents

Interpretation:

- This appears to be a workspace-routing limitation in the query path, not a failure of the OpenAI ingestion itself

### Direct workspace query validation

To isolate the data path from the API workspace issue, the fresh workspace was queried directly through a LightRAG instance configured for:

- workspace `openai_smoke_min1`
- PostgreSQL storages
- `gpt-4.1-mini`
- `text-embedding-3-small`

Query used:

- `What is the Journal of Special Education and Rehabilitation?`

Result:

- Success
- The system retrieved entities, relations, and chunks from the fresh workspace
- It returned a grounded answer with references to the journal and its Wikipedia source

This is the key validation point:

- The OpenAI-backed pipeline produced queryable data in the fresh workspace

## What This Run Demonstrated

The OpenAI path successfully demonstrated:

- phase 1 through phase 4 from scratch,
- phase 6 ingestion into a fresh workspace,
- stable extraction and merge behavior over many documents,
- and successful retrieval when querying the workspace directly.

Compared with the Ollama path, the OpenAI path showed:

- better ingestion stability,
- no observed extraction timeout failures during the tested portion,
- and much better practical suitability for continuing to a real full run.

## Final Conclusion

This smoke test validates the OpenAI path as the correct next direction.

Why:

- The full pipeline can be started from phase 1 successfully.
- The fresh workspace receives valid ingested data.
- Direct retrieval works correctly against that workspace.
- The OpenAI path is materially more stable than the local Ollama path.