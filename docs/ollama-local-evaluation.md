# Local Ollama Evaluation For LightRAG Pipeline

## Purpose

This document records the full local Ollama evaluation that was performed for the Macedonian cultural knowledge pipeline in LightRAG.

The goal was to answer a practical question:

Can the full pipeline be run locally with Ollama in a way that is reliable enough, fast enough, and accurate enough for real use?

Short answer:

- The local Ollama path can work for smoke tests and very small batches.
- It is not a good primary path for full ingestion of this dataset.
- The main reasons are runtime cost, intermittent extraction failures, and retrieval-quality regressions when using the faster summary-based workaround.

## Environment Under Test

The local path used:

- `docker-compose.yml` plus `docker-compose.override.yml`
- Local PostgreSQL via Docker Compose
- Local Ollama server on the host
- LLM: `qwen2.5:7b`
- Embeddings: `bge-m3:latest`
- Workspace: `ollama_exp1`

Relevant local settings at the time of testing:

- `LLM_BINDING=ollama`
- `LLM_MODEL=qwen2.5:7b`
- `LLM_BINDING_HOST=http://host.docker.internal:11434`
- `EMBEDDING_BINDING=ollama`
- `EMBEDDING_MODEL=bge-m3:latest`
- `EMBEDDING_BINDING_HOST=http://host.docker.internal:11434`
- `OLLAMA_LLM_NUM_CTX=8192`
- `LLM_TIMEOUT=600`
- `MAX_ASYNC=1`
- `MAX_PARALLEL_INSERT=1`
- `EMBEDDING_FUNC_MAX_ASYNC=2`

## What Was Checked First

Before running ingestion, the following was verified:

- Docker Compose was available.
- The Ollama models were present locally.
- The API container could be built and started.
- The running API reported healthy status from inside the container.
- The compose stack used PostgreSQL storage successfully.

Observed:

- `qwen2.5:7b` and `bge-m3:latest` were both installed in Ollama.
- The LightRAG API container built successfully.
- The API started successfully and initialized PostgreSQL tables and indexes.
- The running API reported healthy status in workspace `ollama_exp1`.

## Existing Pipeline Data At Start

The repo already contained normalized pipeline artifacts.

The normalized dataset size at the time of testing:

- Documents: `48`
- Total ingest chunks: `523`
- Average chunks per document: `10.9`
- Maximum chunks for a single document: `73`

This is important because local extraction cost scales mostly with the number of extraction chunks, not just the number of documents.

## Problems Found Before Benchmarking

The local Ollama path did not fail only because of model quality. Real integration issues were found in the standalone pipeline path.

### 1. Storage backend mismatch

Problem:

- The standalone pipeline ingestion path could run with file-based default storage.
- The running API was using PostgreSQL-backed storage.
- That meant the pipeline path and the app path were not guaranteed to write to the same backend.

Impact:

- A pipeline run could appear to succeed while the app still queried a different data backend.

Fix applied:

- The standalone pipeline ingestion path was changed to use the same `LIGHTRAG_*_STORAGE` environment-backed storage selection as the API.

### 2. Chunk preservation mismatch

Problem:

- Phase 4 normalization already produced bounded `ingest_chunks`.
- The ingestion path was not fully preserving those prepared chunks for insertion.

Impact:

- The pipeline could pay extra chunking cost and diverge from the intended normalized-ingest structure.

Fix applied:

- Ingestion was changed to preserve the precomputed `ingest_chunks` for structured ingestion.

### 3. Embedding model naming mismatch

Problem:

- The standalone ingestion path did not always attach `model_name` to the embedding wrapper.

Impact:

- PostgreSQL vector table naming and workspace isolation could diverge from the API path.

Fix applied:

- The embedding wrapper was updated to carry the embedding model name consistently.

## Step-By-Step Test Log

### Step 1. Bring up the local stack

Action:

- Built and started the stack with Docker Compose override.

Result:

- Success.
- LightRAG API and PostgreSQL both started successfully.

What this proved:

- The local infrastructure path itself was valid.
- Failures later in the process were not caused by compose startup problems.

### Step 2. Verify health of the running API

Action:

- Queried `/health` from inside the running container.

Result:

- Success.
- The API reported healthy status.
- PostgreSQL-backed storage was active.
- Ollama-backed LLM and embedding bindings were visible in configuration.

What this proved:

- The app runtime was configured correctly enough to begin real ingestion tests.

### Step 3. Benchmark one document using full structured ingestion

Action:

- Ran the ingestion pipeline on one normalized document using structured text.
- The test document was `Q1091325` with `10` ingest chunks.

Initial issue:

- The first run exposed the storage mismatch described above and was corrected.

Final benchmark result after fixes:

- One structured document took `669.2` seconds.
- That is about `11.2` minutes for one 10-chunk document.

What happened during the run:

- The model processed all 10 chunks.
- Extraction and merge completed successfully.
- The document was marked processed in PostgreSQL.

What this proved:

- Full structured local ingestion can succeed.
- It is far too slow for the full dataset.

Projected cost from this benchmark:

- `48` docs at this pace would take roughly `9-10` hours.
- That estimate is before accounting for retries, timeouts, or failed documents.

Conclusion from Step 3:

- Full structured ingestion with local Ollama is not practical for this dataset.

### Step 4. Query the app after single-document ingestion

Action:

- Queried the running LightRAG API for `Who is Zharko Basheski?`

Result:

- Success.
- The app returned a grounded answer with a reference to the ingested Wikipedia source.

What this proved:

- The pipeline and app were connected correctly after the ingestion fixes.
- The local path can support an end-to-end smoke test.

### Step 5. Benchmark one document using summary-backed ingestion

Action:

- Generated an LLM summary for the same document.
- Ingested the summary instead of the full structured multi-chunk text.

Result:

- Summarization took `9.15` seconds.
- Ingestion took `29.32` seconds.
- Total time was `38.47` seconds.

What this proved:

- Summary-backed ingestion is dramatically faster than full structured local ingestion.
- It is the only local path that looked remotely feasible for the full dataset.

### Step 6. Estimate full-dataset runtime under the summary-backed path

Observed basis:

- `48/48` summaries were generated successfully in `371.99` seconds.
- Bulk summary-based ingestion then started.

Rough expectation:

- Total time looked closer to `30-60` minutes rather than `9-10` hours.

Why this mattered:

- This was the only Ollama-based path worth trying for a full run.

### Step 7. Attempt the full dataset using summary-backed ingestion

Action:

- Ran summarization for all normalized docs.
- Then started bulk ingestion into the real app workspace `ollama_exp1`.

Result:

- Summarization phase completed successfully for all `48/48` documents.
- Bulk ingestion made real progress.
- But it did not complete cleanly enough to justify using this path.

Status snapshot during the run:

- Processed docs increased steadily at first.
- The run eventually showed a mix of `processed`, `pending`, `processing`, and `failed`.

Concrete counts after enough evidence was gathered and the ad hoc job was stopped:

- `15 processed`
- `2 failed`
- `31 pending`
- `1 processing`

What this proved:

- The summary-based local path is not dead on arrival.
- But it is still too brittle for dependable full ingestion.

## Failures Observed

### Failure 1. Ollama extraction timeout during bulk ingestion

Concrete failed document:

- `Q6723101`
- Label: `North Macedonia national under-17 football team`

Observed failure:

- Extraction failed with `httpx.ReadTimeout`

What this means:

- The model/backend path was not just slow.
- It was unstable under repeated extraction load.

Why this matters:

- A long-running local batch job must be stable, not just theoretically capable.
- A pipeline that produces intermittent extraction timeouts is not reliable enough for full ingestion without manual cleanup and retry logic.

### Failure 2. Duplicate record during repeated testing

Observed failed row:

- Duplicate of `Q1091325`

Cause:

- This happened because the document had already been ingested during an earlier benchmark.

Interpretation:

- This was expected test-state behavior, not a model failure.

### Failure 3. Summary-backed retrieval quality regression

Observed behavior:

- A newly summary-ingested document was not found in `mix` mode.
- The same document could be found in `naive` mode.

Example:

- Querying for `Journal of Special Education and Rehabilitation` in `mix` mode produced no relevant context.
- Retrying in `naive` mode did retrieve the document.

What this means:

- The summary-based workaround improves speed.
- But it weakens the richness of the extracted graph and retrieval path.

Why this matters:

- The point of this system is not just to ingest faster.
- It is to ingest in a way that preserves retrieval quality, especially for KG-aware query modes.

## Why The Ollama Path Is Ultimately Not Worth It

This decision was not based on one issue. It was based on the combination below.

### 1. Full structured ingestion is too slow

Measured result:

- `~11.2` minutes for one 10-chunk document

Projected result:

- `~9-10` hours for the full dataset

Conclusion:

- This is not a practical local workflow for iterative development or project delivery.

### 2. The faster workaround reduces quality

The only locally viable path was summary-backed ingestion.

But that workaround:

- compresses the input heavily,
- reduces extracted entity/relation richness,
- and can cause retrieval misses in KG-aware query modes.

Conclusion:

- The workaround makes the system faster, but less faithful to the actual target behavior.

### 3. The local backend is not stable enough under batch extraction

Observed:

- Bulk ingestion hit `httpx.ReadTimeout` against local Ollama.
- The run continued, but not cleanly enough to trust for full-corpus operation.

Conclusion:

- Even if the speed is barely acceptable with summaries, the error profile is still too fragile.

### 4. The resulting system would still be weaker at query time

Even if ingestion were forced to complete locally:

- retrieval quality would likely be worse,
- KG extraction would likely be thinner,
- and query behavior in `mix`/graph-aware modes would be less dependable.

Conclusion:

- This is not just an ingestion-time problem.
- It affects the usefulness of the final system.

## Final Decision

The local Ollama path is acceptable only for:

- smoke tests,
- debugging the pipeline shape,
- validating small-batch ingestion,
- and proving the app can run locally at all.

It is not recommended for:

- full dataset ingestion,
- final evaluation,
- or production-like quality testing of the retrieval pipeline.

## Recommended Next Path

Move to an OpenAI-backed configuration for:

- LLM extraction,
- summarization if needed,
- and embeddings.

Reason:

- It avoids the local throughput bottleneck.
- It reduces timeout instability.
- It preserves better retrieval quality than the summary-only local workaround.

## Bottom-Line Summary

The Ollama path was tested seriously and not dismissed prematurely.

What was proven:

- The stack runs locally.
- End-to-end ingestion and querying work on small cases.
- The summary-backed workaround can speed ingestion substantially.

What was also proven:

- Full structured local ingestion is too slow.
- The faster summary-only workaround hurts retrieval quality.
- Local Ollama extraction suffers intermittent timeout failures.

Final recommendation:

- Do not use the Ollama path as the primary ingestion strategy for this project.
- Use it only for local smoke testing.
- Use the OpenAI path for the actual full-pipeline workflow.
