# RAG Evaluation Report

## Dataset

- Dataset: `lightrag/evaluation/vezilka_qa_ragas_dataset.json`
- QA cases: 100
- Evaluated rows: 300
- Evaluated modes: naive, mix, hybrid
- Evaluated systems: graph, mixed, naive

## Metrics

- RAGAS: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness, answer_similarity when RAGAS and evaluator LLM credentials are available.
- Retrieval: Precision@K, Recall@K, MRR, NDCG@K when ground-truth source ids exist; average retrieved chunks and empty retrieval rate always run.
- Generation: exact match, token-level F1, lexical semantic similarity fallback. LLM judge correctness is TODO unless an evaluator judge is configured.
- System: average latency, median latency, total time, failures, token estimate.
- Graph: exposed entity/relationship counts are recorded. Depth/hops, graph coverage, and full entity recall require traversal metadata and ground-truth graph entities.

RAGAS status: Computed RAGAS metrics using top 5 retrieved contexts per row, 1200 chars/context, batch_size=1, max_workers=1; raw scores written to evaluation_results_full_ragas_fixed/ragas_raw_scores.csv. Missing metric values: 0.

## Comparison

| Metric | graph | mixed | naive |
|---|---:|---:|---:|
| Faithfulness | 0.8426 | 0.8396 | 0.8720 |
| Answer Relevancy | 0.6742 | 0.6685 | 0.6745 |
| Context Precision | 0.6350 | 0.6482 | 0.6688 |
| Context Recall | 0.8367 | 0.8033 | 0.8367 |
| Answer Correctness | 0.4337 | 0.4731 | 0.4721 |
| Token F1 | 0.1063 | 0.1055 | 0.1113 |
| Semantic Similarity | 0.0760 | 0.0741 | 0.0781 |
| Precision@K | 0.5820 | 0.5720 | 0.6320 |
| Recall@K | 0.9467 | 0.9433 | 0.9500 |
| MRR | 0.9333 | 0.9358 | 0.9458 |
| NDCG@K | 1.6816 | 1.6714 | 1.7754 |
| Latency seconds | 3.2629 | 3.1552 | 0.6150 |
| Failed queries | 0 | 0 | 0 |

## Best Examples

### VZ-QA-069 / naive

- Question: Summarize Jasin Ferati using both its corpus description and one detailed supporting fact.
- Expected answer: Jasin Ferati — Swiss racing driver. Supporting detail: He is the 2022 Porsche Sports Cup Suisse champion, having previously competed in the Formula Regional European Championship.
- Generated answer: Jasin Ferati is a Swiss racing driver born on August 2, 2003, in Winterthur. He has Macedonian Albanian heritage, with his parents originating from Gostivar. Ferati began his karting career in 2016 and steadily progressed through the ranks, eventually competing in prominent competitions such as the Formula Regional European Championship and the Porsche Sports Cup Suisse. In 2022, he won the Porsche Sports Cup Suisse GT3 Cup premier class title, securing six race wins out of 12 races in a competitive season against series veteran Jürg Aeberhard. This achievement marked a significant highlight of his racing career as he transitioned to sportscar racing full-time [1]. ### References - [1] Jasin Ferati | Q106634862 | en.wikipedia | https://en.wikipedia.org/wiki/Jasin_Ferati
- Retrieved contexts summary: Jasin Ferati is a човек. Jasin Ferati was born on 2003-08-02. Jasin Ferati was born in Винтертур. Jasin Ferati is a citizen of Швајцарија, Албанија and Македонија. Jasin Ferati's occupation: мототркач.
- Score signal: token_f1=0.2639, faithfulness=1.0000, answer_correctness=0.9674
- Explanation: Higher scores indicate closer answer overlap and, when available, stronger RAGAS grounding/correctness. Review the retrieved context to decide whether retrieval or generation caused the gap.

### VZ-QA-028 / graph

- Question: Which English/Latin entity label corresponds to the Macedonian label Црква „Пресвето Срце Исусово“ - Скопје?
- Expected answer: Црква „Пресвето Срце Исусово“ - Скопје corresponds to Cathedral of the Sacred Heart of Jesus.
- Generated answer: The English/Latin entity label corresponding to the Macedonian label Црква „Пресвето Срце Исусово“ - Скопје is "Cathedral of the Sacred Heart of Jesus." ### References - [1] Црква „Пресвето Срце Исусово“ - Скопје | Q11737299 | mk.wikipedia | https://mk.wikipedia.org/wiki/%D0%A6%D1%80%D0%BA%D0%B2%D0%B0_%E2%80%9E%D0%9F%D1%80%D0%B5%D1%81%D0%B2%D0%B5%D1%82%D0%BE_%D0%A1%D1%80%D1%86%D0%B5_%D0%98%D1%81%D1%83%D1%81%D0%BE%D0%B2%D0%BE%E2%80%9C_-_%D0%A1%D0%BA%D0%BE%D0%BF%D1%98%D0%B5
- Retrieved contexts summary: Entity aliases (same entity): Macedonian/Cyrillic: Црква „Пресвето Срце Исусово“ - Скопје; English/Latin: Cathedral of the Sacred Heart of Jesus.
- Score signal: token_f1=0.2185, faithfulness=1.0000, answer_correctness=0.9307
- Explanation: Higher scores indicate closer answer overlap and, when available, stronger RAGAS grounding/correctness. Review the retrieved context to decide whether retrieval or generation caused the gap.

### VZ-QA-030 / graph

- Question: Which English/Latin entity label corresponds to the Macedonian label Борис Чипан?
- Expected answer: Борис Чипан corresponds to Boris Čipan.
- Generated answer: The English/Latin entity label that corresponds to the Macedonian label Борис Чипан is Boris Čipan. ### References - [2] Борис Чипан | Q12273756 | mk.wikipedia | https://mk.wikipedia.org/wiki/%D0%91%D0%BE%D1%80%D0%B8%D1%81_%D0%A7%D0%B8%D0%BF%D0%B0%D0%BD
- Retrieved contexts summary: Ilhami Emin — Macedonian-Turkish poet, translator.
- Score signal: token_f1=0.2222, faithfulness=1.0000, answer_correctness=0.9394
- Explanation: Higher scores indicate closer answer overlap and, when available, stronger RAGAS grounding/correctness. Review the retrieved context to decide whether retrieval or generation caused the gap.


## Worst Examples

### VZ-QA-098 / mixed

- Question: Name three Vezilka corpus entities with instance type музеј and state the core fact for each.
- Expected answer: Museum of Skopje Old Bazaar is represented as Museum of Skopje Old Bazaar — museum in Skopje.; Porta Macedonia is represented as Porta Macedonia — triumphal arch in Skopje, North Macedonia.; National Gallery of North Macedonia is represented as National Gallery of North Macedonia — museum in Macedonia.
- Generated answer: Three Vezilka corpus entities with the instance type музеј (museum) along with their core facts are: 1. Музеј на Град Скопје (Museum of the City of Skopje) - Core fact: It is defined explicitly as a музеј (museum). 2. Порта Македонија (Porta Macedonia) - Core fact: Classified as a музеј (museum). 3. Завод и музеј (Zavod i Muzej) - Core fact: It is an organization that includes the Jeni Mosque as part of its structure. ### References - [1] Музеј на Град Скопје defined as музеј - [2] Порта Македонија classified as музеј - [3] Јени џамија is part of Завод и музеј
- Retrieved contexts summary: Entity aliases (same entity): Macedonian/Cyrillic: Вера Чејковска; English/Latin: Vera Chejkovska.
- Score signal: token_f1=0.2411, faithfulness=0.0000, answer_correctness=0.1409
- Explanation: Higher scores indicate closer answer overlap and, when available, stronger RAGAS grounding/correctness. Review the retrieved context to decide whether retrieval or generation caused the gap.

### VZ-QA-093 / naive

- Question: Name three Vezilka corpus entities with instance type човек and state the core fact for each.
- Expected answer: Ljubiša Georgievski is represented as Ljubiša Georgievski — Macedonian politician (1937-2018).; Jasin Ferati is represented as Jasin Ferati — Swiss racing driver.; Puleksenija Jovanoska is represented as Puleksenija Jovanoska — North Macedonian karateka.
- Generated answer: Here are three entities from the Vezilka corpus with the instance type човек (person), along with a core fact for each: 1. Врчак (Vrčak) Core fact: Vrčak is recognized as an individual person entity. Reference: [1] 2. Вера Чејковска (Vera Chejkovska) Core fact: Vera Chejkovska is also categorized as a person. Reference: [2] 3. Драган Вучиќ (Dragan Vučić) Core fact: Dragan Vučić is identified as a human entity. Reference: [4] ### References - [1] Врчак | Q956646 | mk.wikipedia | https://mk.wikipedia.org/wiki/%D0%92%D1%80%D1%87%D0%B0%D0%BA - [2] Вера Чејковска | Q59946 | mk.wikipedia | https://mk.wikipedia.org/wiki/%D0%92%D0%B5%D1%80%D0%B0_%D0%A7%D0%B5%D1%98%D0%BA%D0%BE%D0%B2%D1%81%D0%BA%D0%B0 - [4] Драган Вучиќ | Q5304908 | mk.wikipedia | https://mk.wikipedia.org/wiki/%D0%94%D1%80%D0%B0%D0%B3%D0%B0%D0%BD_%D0%92%D1%83%D1%87%D0%B8%D1%9C
- Retrieved contexts summary: Entity aliases (same entity): Macedonian/Cyrillic: Врчак; English/Latin: Vrčak.
- Score signal: token_f1=0.0625, faithfulness=0.4286, answer_correctness=0.0794
- Explanation: Higher scores indicate closer answer overlap and, when available, stronger RAGAS grounding/correctness. Review the retrieved context to decide whether retrieval or generation caused the gap.

### VZ-QA-015 / graph

- Question: What is Синагога Арагон (Битола) in the Vezilka corpus?
- Expected answer: Синагога Арагон (Битола) is described as поранешна синагога во Битола; its instance type is синагога.
- Generated answer: Синагога Арагон (Битола) во Vezilka корпусот е позната како поранешна синагога сместена во Општина Битола, Македонија. Таа претставува синагога која била централно место за молитва на Евреите од Арагонскиот (шпански) потекло во Битола, особено во 16 век. Првично, во неа се собирале еврејски имигранти од Португалија за молитва пред да изградат сопствена синагога позната како Синагога Португезе, која се наоѓала спроти веќе постоечката Арагон синагога. Оваа синагога била една од најголемите во Балканскиот регион и имала Torah Ark со свиток на Тора. За време на Втората светска војна, бугарската фашистичка армија ја зазела и ја претворила во кланица за свињи. Синагогата била изгорена малку пред крајот на војната. Неколку значајни рабини служеле во оваа синагога во различни периоди, меѓу кои се споменуваат имиња како Аврахам Фалкон, Меркадо Камхи и други. Овој објект е дел од културното наследство на Македонија и се наоѓа во општествената и историската рамка на градот Битола. (Референца: [1]) ### References - [1] Синагога Арагон (Битола) | Q114245520 | mk.wikipedia | https://mk.wikipedia.org/wiki/%D0%A1%D0%B8%D0%BD%D0%B0%D0%B3%D0%BE%D0%B3%D0%B0_%D0%90%D1%80%D0%B0%D0%B3%D0%BE%D0%BD_%28%D0%91%D0%B8%D1%82%D0%BE%D0%BB%D0%B0%29
- Retrieved contexts summary: Синагога Арагон (Битола) is a синагога. Синагога Арагон (Битола) is associated with Македонија. Синагога Арагон (Битола) is located in Општина Битола.
- Score signal: token_f1=0.0721, faithfulness=0.1875, answer_correctness=0.3036
- Explanation: Higher scores indicate closer answer overlap and, when available, stronger RAGAS grounding/correctness. Review the retrieved context to decide whether retrieval or generation caused the gap.


## Failed Examples

- No failed examples.

## Limitations

- RAGAS metrics are skipped if `ragas`, `datasets`, `langchain-openai`, or evaluator API keys are missing.
- Precision@K/Recall@K/MRR/NDCG@K require source ids in the QA data and comparable ids in retrieved contexts.
- Token usage is estimated unless the LightRAG API returns provider usage metadata.
- Judge-based correctness is not computed unless an LLM judge is explicitly added.
- Graph depth/hops and graph coverage need LightRAG to return traversal paths and ground-truth graph nodes/entities in the dataset.

## Output Files

- per_question_results.csv: `evaluation_results_full_ragas_fixed/per_question_results.csv`
- per_question_results.json: `evaluation_results_full_ragas_fixed/per_question_results.json`
- summary_metrics.csv: `evaluation_results_full_ragas_fixed/summary_metrics.csv`
- summary_metrics.json: `evaluation_results_full_ragas_fixed/summary_metrics.json`
- qa_comparison_wide.csv: `evaluation_results_full_ragas_fixed/qa_comparison_wide.csv`
- evaluation_report.md: `evaluation_results_full_ragas_fixed/evaluation_report.md`
