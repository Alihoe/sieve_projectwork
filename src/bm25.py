from rank_bm25 import BM25Okapi
import numpy as np


def bm25_rank(queries, data, threshold=None, top_k=None):
    data_items = list(data.items())
    data_ids = [x[0] for x in data_items]
    data_texts = [x[1] for x in data_items]
    tokenized_data = [doc.split() for doc in data_texts]

    bm25 = BM25Okapi(tokenized_data)

    ranked_results = {}
    not_ranked = []

    for query_id, query_text in queries.items():
        tokenized_query = query_text.split()
        scores = bm25.get_scores(tokenized_query)
        scored_docs = sorted(zip(data_ids, scores), key=lambda x: x[1], reverse=True)

        if threshold is not None:
            scored_docs = [(doc_id, score) for doc_id, score in scored_docs if score >= threshold]

        if scored_docs:
            if top_k:
                ranked_results[query_id] = [doc_id for doc_id, score in scored_docs[:top_k]]
            else:
                ranked_results[query_id] = [doc_id for doc_id, score in scored_docs]
        else:
            not_ranked.append(query_id)

    return ranked_results, not_ranked


def calculate_threshold(queries, data, gold_data, min_similarity=0.3):
    data_items = list(data.items())
    data_ids = [x[0] for x in data_items]
    data_texts = [x[1] for x in data_items]
    tokenized_data = [doc.split() for doc in data_texts]

    bm25 = BM25Okapi(tokenized_data)
    data_id_to_idx = {data_id: idx for idx, data_id in enumerate(data_ids)}
    relevant_scores = []

    for query_id, query_text in queries.items():
        if query_id not in gold_data:
            continue

        tokenized_query = query_text.split()
        scores = bm25.get_scores(tokenized_query)
        gold_indices = [data_id_to_idx[doc_id] for doc_id in gold_data[query_id] if doc_id in data_id_to_idx]
        relevant_scores.extend(scores[gold_indices])

    if not relevant_scores:
        return min_similarity

    mean_rel = np.mean(relevant_scores)
    std_rel = np.std(relevant_scores)
    return max(min_similarity, mean_rel + std_rel)