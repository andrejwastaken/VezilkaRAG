# LightRAG Mode Evaluation Explanation

## What Was Tested

This evaluation tested the Vezilka LightRAG setup on 100 QA cases. Each question was run through three LightRAG modes, producing 300 evaluated answers:

| Evaluation label | LightRAG mode | Rows | Meaning |
|---|---|---:|---|
| naive | naive | 100 | Direct chunk/vector-style retrieval baseline. |
| graph | mix | 100 | LightRAG graph-oriented retrieval using entities, relationships, and related chunks. |
| mixed | hybrid | 100 | Hybrid retrieval combining graph signals with direct retrieval. |

The QA dataset includes several query types:

| Query type | Count |
|---|---:|
| single-hop / targeted factual | 35 |
| single-hop / single-chunk factual | 18 |
| single-hop / multi-chunk synthesis | 15 |
| single-hop bilingual alias | 12 |
| multi-hop / cross-document comparison | 12 |
| multi-hop / set retrieval | 8 |

The evaluation compared each generated answer against the expected answer and retrieved context using:

- RAGAS generation and grounding metrics: faithfulness, answer relevancy, context precision, context recall, answer correctness, answer similarity.
- Retrieval metrics: Precision@K, Recall@K, MRR, NDCG@K.
- Local generation metrics: exact match, token F1, lexical semantic similarity.
- System metrics: latency, retrieved chunk count, failed query count.
- Graph metadata where available: graph node count and graph edge/relationship count.

## Main Results

| Metric | naive | graph | mixed |
|---|---:|---:|---:|
| RAGAS score | 0.6985 | 0.6819 | 0.6836 |
| Faithfulness | 0.8720 | 0.8426 | 0.8396 |
| Answer relevancy | 0.6745 | 0.6742 | 0.6685 |
| Context precision | 0.6688 | 0.6350 | 0.6482 |
| Context recall | 0.8367 | 0.8367 | 0.8033 |
| Answer correctness | 0.4721 | 0.4337 | 0.4731 |
| Answer similarity | 0.6671 | 0.6695 | 0.6686 |
| Precision@K | 0.6320 | 0.5820 | 0.5720 |
| Recall@K | 0.9500 | 0.9467 | 0.9433 |
| MRR | 0.9458 | 0.9333 | 0.9358 |
| Avg retrieved chunks | 5.00 | 39.57 | 39.84 |
| Avg graph nodes | 0.00 | 11.05 | 11.03 |
| Avg graph edges | 0.00 | 23.51 | 23.80 |
| Avg latency seconds | 0.6150 | 3.2629 | 3.1552 |
| Failed queries | 0 | 0 | 0 |

All 300 rows completed successfully. All RAGAS metric cells were populated, with 0 missing RAGAS values.

## What The Differences Mean

The naive mode performed best overall in this dataset. It had the highest average RAGAS score, faithfulness, context precision, Precision@K, Recall@K, MRR, token F1, and the lowest latency.

This does not mean graph retrieval is useless. It means that for this particular QA set, direct retrieval was usually enough. Most questions are factual or entity-centered, and many expected answers can be found in a small number of directly relevant chunks. In that setup, naive retrieval has two advantages:

- It retrieves fewer chunks, so the answer generator receives less distracting context.
- It is faster because it does not expand through graph entities and relationships.

Graph mode retrieved far more context: about 39.57 chunks on average versus 5.00 for naive. It also retrieved about 11 graph nodes and 23.5 graph relationships per question. That broader context can help when a question requires entity linking, related facts, or multi-hop reasoning, but it can also add noise. The lower context precision for graph mode suggests that some of the additional graph-expanded material was not directly needed for the expected answer.

Mixed mode was close to graph mode overall. It had slightly better answer correctness than graph mode and slightly better context precision, but lower context recall. This suggests the hybrid strategy sometimes filters or balances graph expansion better than pure graph-oriented retrieval, but it still carries much of the same extra-context cost.

## Why Naive Won Here

The dataset is mostly short factual QA. Even though it includes multi-hop and multi-chunk categories, the ground-truth answers are usually concise and tied to known source documents. Because the source IDs and labels are strongly represented in the text chunks, direct retrieval can often find the right evidence without graph traversal.

Naive mode also produced the smallest context set. Smaller context sets often improve faithfulness when the correct evidence is already present, because the model has fewer unrelated facts to reconcile.

Graph and mixed modes are expected to show more advantage on questions where:

- the answer depends on relationships between multiple entities,
- the query uses aliases or indirect references,
- the relevant evidence is not lexically similar to the question,
- the answer requires following graph links across documents,
- source chunks alone are sparse or ambiguous.

This QA set contains some of those questions, but not enough for graph expansion to beat the direct retrieval baseline on average.

## Graph-Specific Metrics

Two graph metadata fields were available and populated:

- `graph_nodes_retrieved`: number of graph entities returned by LightRAG `/query/data`.
- `graph_edges_traversed`: number of graph relationships returned by LightRAG `/query/data`.

Three deeper graph metrics were not populated in the completed result:

- `graph_average_depth_hops`
- `graph_coverage`
- `entity_recall`

### Can These Be Implemented?

`entity_recall` can be partially implemented now. The dataset has `source_documents` with `qid`, `label_en`, and `label_mk`. The evaluator has been updated so future runs derive expected entities from those fields. That allows checking whether expected entity labels or QIDs appear in retrieved contexts or retrieved IDs.

`graph_coverage` can be implemented properly only if LightRAG returns the actual graph node IDs or labels that were retrieved. The current saved output has node counts, but not the full list of returned graph node identifiers. A count alone is not enough to know whether the expected nodes were covered.

`graph_average_depth_hops` requires traversal path metadata from LightRAG. The current API response exposes entities and relationships, but not the traversal path or hop depth used to reach them. To compute this correctly, LightRAG would need to return fields such as:

- seed entity or starting chunk,
- retrieved graph node IDs,
- relationship path IDs,
- hop distance for each node,
- path from query entity to retrieved entity.

Without that path metadata, any depth/hops value would be an approximation and should not be reported as a real graph metric.

## Conclusion

For the current Vezilka QA dataset, naive retrieval is the strongest overall mode. It is also much faster. The main reason is that the questions are often answerable from directly retrieved source chunks, so graph expansion adds more context than necessary.

Graph and mixed modes are functioning: they return graph nodes and relationships, retrieve relevant material, and produce competitive answers. However, their extra context appears to reduce precision and increase latency for this dataset.

The practical conclusion is:

- Use naive mode as the current baseline and likely default for short factual questions.
- Use graph or mixed mode for questions that need entity relationships, cross-document reasoning, aliases, or multi-hop retrieval.
- Improve graph evaluation by exposing graph traversal metadata from LightRAG and adding expected graph entities/nodes to the QA dataset.

The evaluation does not show a broken graph pipeline. It shows that the current test set favors direct retrieval more than graph expansion.
