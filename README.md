# Scientific Claim Resource Retrieval

This repository contains my work for **CLEF 2025 Task 4, Subtask 4b: Scientific Claim Source Retrieval**.

In this task, the goal is to retrieve the scientific paper referenced by a social media post that mentions a publication implicitly, without linking to it directly.

## Task

Given an implicit reference to a scientific paper, that is, a tweet mentioning a research publication without a URL, retrieve the corresponding paper from a pool of candidate papers.

The official task description is also included in [the shared CLEF materials](C:\repos\scientific_claim_resource_retrieval\clef2025-checkthat-lab-main\task4\subtask_4b\README.md).

## System Overview

My retrieval system, **Sieve**, is built as a pipeline of retrieval components ranging from low computational cost to higher computational cost. Queries that cannot be predicted confidently by an earlier stage are passed to the next component. In addition, some queries are routed through data-specific components when they contain attributes such as direct quotes, journal mentions, or author names.

The overall design goal was to resolve easy cases cheaply and reserve more expensive semantic retrieval and language-model-based ranking for harder cases.

## Pre-Processing

The collection is loaded from `data/subtask4b_collection_data.pkl`. The code builds two text views of each paper:

- `title + abstract`
- `title + summary`

If an abstract has fewer than 100 characters, the system falls back to the title. This behavior is implemented in [`src/utils.py`](C:\repos\scientific_claim_resource_retrieval\src\utils.py).

I also generated three-sentence summaries of abstracts using `llama3.1:8b`. The summarization model is defined in [`src/summarize.py`](C:\repos\scientific_claim_resource_retrieval\src\summarize.py). In the retrieval pipeline, full abstracts are used mainly for BM25-style lexical matching, while summaries are used for the semantic similarity components.

## Pipeline Components

The repository contains multiple experimental scripts, but the committed final pipeline is represented most clearly in [`final_sievetrival.py`](C:\repos\scientific_claim_resource_retrieval\final_sievetrival.py). Its active component order is:

1. Title Detection
2. Quote Detection
3. BM25 on Abstracts
4. Semantic Similarity with MiniLM
5. Semantic Similarity with MPNet
6. Semantic Similarity with SPECTER
7. Journal Detection
8. Author Detection
9. Lower-threshold MiniLM fallback
10. No-threshold semantic fill step with BGE
11. Final no-threshold semantic retrieval for unresolved cases

The final script truncates each prediction list to top 5 and writes `PREDICTIONS.tsv`.

### Title Detection

This is a direct string matching component implemented in [`src/title_matching.py`](C:\repos\scientific_claim_resource_retrieval\src\title_matching.py). For each query, the system checks whether the lowercase title of any paper is a substring of the lowercase query text. If one or more titles are found, those paper IDs are returned.

### Quote Detection

Quote detection is implemented in [`src/quote_detection.py`](C:\repos\scientific_claim_resource_retrieval\src\quote_detection.py). The method:

- repairs missing whitespace around punctuation
- extracts quoted spans with regular expressions
- keeps only quotes longer than three words
- removes references from full paper text to reduce noise
- matches extracted quotes against the full paper text or the paper title

If fewer than four candidate papers are found, those candidates are returned.

### Journal Detection

Journal matching is implemented in [`src/journal_matching.py`](C:\repos\scientific_claim_resource_retrieval\src\journal_matching.py). It checks whether the query contains an exact journal name from the collection, using regex matching to avoid partial-word false positives. It also looks for lead-in phrases such as `published in` and `released in`. To reduce false positives, the code filters journal names against a large common-word list and a custom stop list.

If a journal is identified, all papers from that journal are returned as candidates.

### Author Detection

Author matching is implemented in [`src/author_matching.py`](C:\repos\scientific_claim_resource_retrieval\src\author_matching.py). For each author, the code generates multiple name variants such as:

- `John Doe`
- `Doe John`
- `Doe`

The query is normalized, punctuation is removed, and author variants are matched as standalone spans. If several matched author names point to the same paper, the candidate set is narrowed by set intersection.

### BM25

BM25 retrieval is implemented in [`src/bm25.py`](C:\repos\scientific_claim_resource_retrieval\src\bm25.py) and [`src/bm252.py`](C:\repos\scientific_claim_resource_retrieval\src\bm252.py). In the final pipeline, BM25 is used on the abstract-based representation to capture strong lexical overlap.

### Semantic Similarity

Semantic similarity ranking is implemented in [`src/sentence_similarity.py`](C:\repos\scientific_claim_resource_retrieval\src\sentence_similarity.py). I experimented with several embedding models, including:

- `all-MiniLM-L6-v2`
- `all-mpnet-base-v2`
- `allenai-specter`
- `BAAI/bge-large-en-v1.5`
- `thenlper/gte-large`
- `intfloat/e5-large-v2`
- `sentence-t5-xxl`

The models that mattered most in the final pipeline were MiniLM, MPNet, SPECTER, and BGE. BGE is used as the final no-threshold fill model in `final_sievetrival.py`.

### LLM Ranking

LLM-based reranking is implemented in [`src/llm_re_ranking.py`](C:\repos\scientific_claim_resource_retrieval\src\llm_re_ranking.py) and uses `llama3.1:8b`.

The design intention was to use a generative language model to rerank candidate lists for the hardest cases. In the committed code, reranking is still applied to some small candidate sets produced by title, quote, journal, and author matching. However, the large final reranking step is commented out in the final scripts, which matches the project note that this step was not completed in time for the final submission.

## Thresholds and Active Settings

The repository contains several thresholding experiments. The code does compute training-based similarity statistics, but the active values in the final scripts are hard-coded. This means the implementation is slightly more specific than the general description of using `average minus one standard deviation` for every component.

In the active `final_sievetrival.py` configuration:

- `n_candidates = 30`
- BM25 abstracts threshold: `61.071572593218015`
- MiniLM threshold: similarity `0.6517946525885147`, distance `0.10337032085336031`
- MPNet threshold: similarity `0.6848235173594495`, distance `0.09782780249794855`
- SPECTER threshold: similarity `0.8423419033753903`, distance `0.041100625714210616`
- BGE is used without thresholds for the fill step
- There is also a lower-threshold MiniLM fallback with similarity `0.65` and distance `0`

Other scripts in the repository test more permissive or alternative settings, including variants with and without reranking.

## Additional Methods I Explored

These components are present in the repository but did not contribute strong enough results to remain central in the final pipeline:

### Numerical Information Extraction

Implemented in [`src/numerical_information.py`](C:\repos\scientific_claim_resource_retrieval\src\numerical_information.py). The method extracts numbers from both digits and number words and ranks documents by overlap in numerical content. This seemed promising for highly quantitative tweets, but it did not generalize well enough across the full task.

### Unique Token Linking

Implemented in [`src/token_matching.py`](C:\repos\scientific_claim_resource_retrieval\src\token_matching.py). This component tries to match tweets and papers by rare overlapping content words, especially nouns and proper nouns. It became less useful once the semantic similarity stack was improved.

### Named Entity Linking

There is also an entity-based experiment in [`src/named_entity_ranking.py`](C:\repos\scientific_claim_resource_retrieval\src\named_entity_ranking.py). The main idea was to exploit rare named entities and linked concepts, but the stronger embedding-based methods reduced its practical impact.

## What I Learned

My final ranking stage did not perform as well as I wanted. The main reason was time: I explored too many settings and did not finish the large-scale LLM reranking experiments in time. I did not include the full final reranking step in the submission.

The main follow-up directions are:

- use generative LLMs more efficiently, ideally on an external server
- compare open-source rerankers with paid models more systematically
- evaluate current embedding models more directly for this task
- determine how many candidates an LLM can rerank effectively before quality drops

