from rank_bm25 import BM25Okapi
import numpy as np
import json


def preprocess(text):
    return text.lower().split()


def calculate_threshold(queries_dict, data_dict, gold_dict):
    data_texts = []
    data_ids = list(data_dict.keys())

    for data_id in data_ids:
        data_texts.append(preprocess(data_dict[data_id]))

    bm25 = BM25Okapi(data_texts)
    all_scores = []

    for query_id in gold_dict:
        query_text = queries_dict[query_id]
        tokenized_query = preprocess(query_text)
        doc_scores = bm25.get_scores(tokenized_query)

        correct_data_id = gold_dict[query_id]
        correct_index = data_ids.index(correct_data_id)
        correct_score = doc_scores[correct_index]
        all_scores.append(correct_score)

    mean = np.mean(all_scores)
    std = np.std(all_scores)
    return mean, std


def rank_queries(queries_dict, data_dict, predictions_dict, threshold):
    data_texts = []
    data_ids = list(data_dict.keys())

    for data_id in data_ids:
        data_texts.append(preprocess(data_dict[data_id]))

    bm25 = BM25Okapi(data_texts)
    ranked_predictions = {}
    not_ranked = []

    for query_id in predictions_dict:
        query_text = queries_dict[query_id]
        tokenized_query = preprocess(query_text)
        doc_scores = bm25.get_scores(tokenized_query)

        data_ids_to_rank = predictions_dict[query_id]
        data_indices = [data_ids.index(d_id) for d_id in data_ids_to_rank]
        relevant_scores = [doc_scores[i] for i in data_indices]

        if max(relevant_scores) > threshold:
            ranked_data = [d_id for _, d_id in sorted(zip(relevant_scores, data_ids_to_rank), reverse=True)]
            ranked_predictions[query_id] = ranked_data
        else:
            not_ranked.append(query_id)

    return ranked_predictions, not_ranked


def rank_all_queries(queries_dict, data_dict, threshold, top_k=5):
    data_texts = []
    data_ids = list(data_dict.keys())

    for data_id in data_ids:
        data_texts.append(preprocess(data_dict[data_id]))

    bm25 = BM25Okapi(data_texts)
    ranked_predictions = {}
    not_ranked = []

    for query_id, query_text in queries_dict.items():
        tokenized_query = preprocess(query_text)
        doc_scores = bm25.get_scores(tokenized_query)

        max_score = max(doc_scores)
        if max_score > threshold:
            ranked_data = [d_id for _, d_id in sorted(zip(doc_scores, data_ids), reverse=True)]
            ranked_predictions[query_id] = ranked_data[:top_k]
        else:
            not_ranked.append(query_id)

    return ranked_predictions, not_ranked

