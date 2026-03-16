import numpy as np
from sentence_transformers import SentenceTransformer, util


MODELS = {
    "mini": "sentence-transformers/all-MiniLM-L6-v2",
    "specter": "sentence-transformers/allenai-specter",
    "scibert": "allenai/scibert_scivocab_uncased",
    "mpnet": "all-mpnet-base-v2",
    "bge": "BAAI/bge-large-en-v1.5",
    "gte": "thenlper/gte-large",
    "e5": "intfloat/e5-large-v2",
    "t5": "sentence-transformers/sentence-t5-xxl",
}


def _load_model(model_name):
    return SentenceTransformer(MODELS[model_name])


def _prepare_text(text, model_name, prefix_type):
    if model_name != "e5":
        return text
    prefix = "query" if prefix_type == "query" else "passage"
    return f"{prefix}: {text}"


def get_similarity_threshold(queries, gold_ids, data_dict, column="summary", model="mini"):
    encoder = _load_model(model)
    similarities = []
    for index, query in enumerate(queries):
        embeddings = encoder.encode(
            [
                _prepare_text(query, model, "query"),
                _prepare_text(data_dict[gold_ids[index]][column], model, "passage"),
            ],
            convert_to_tensor=True,
        )
        similarities.append(util.pytorch_cos_sim(embeddings[0], embeddings[1]).item())
    return np.mean(similarities), np.std(similarities)


def get_distance_threshold(queries, abstracts, data_ids, gold_ids, model="mini"):
    encoder = _load_model(model)
    encoded_abstracts = [_prepare_text(abstract, model, "passage") for abstract in abstracts]
    reference_embeddings = encoder.encode(encoded_abstracts)
    distances = []

    for index, query in enumerate(queries):
        query_embedding = encoder.encode(_prepare_text(query, model, "query"))
        similarities = encoder.similarity(query_embedding, reference_embeddings).tolist()[0]
        top_indices = np.argsort(-np.array(similarities))[:5]
        top_ids = [data_ids[item] for item in top_indices]
        if top_ids and gold_ids[index] == top_ids[0]:
            sorted_similarities = np.sort(similarities)[::-1]
            if len(sorted_similarities) > 1:
                distances.append(sorted_similarities[0] - sorted_similarities[1])

    if not distances:
        return 0.0, 0.0
    return np.mean(distances), np.std(distances)


def rank_to_one_query_semantic_similarity(query, data_ids, data_dict, model="mpnet", top_n=30):
    encoder = _load_model(model)
    data_texts = [
        _prepare_text(f"{data_dict[data_id]['title']}: {data_dict[data_id]['summary']}", model, "passage")
        for data_id in data_ids
    ]
    reference_embeddings = encoder.encode(data_texts)
    query_embedding = encoder.encode(_prepare_text(query, model, "query"))
    similarities = encoder.similarity(query_embedding, reference_embeddings).tolist()[0]
    top_indices = np.argsort(-np.array(similarities))[:top_n]
    return [data_ids[item] for item in top_indices]


def rank_by_semantic_similarity(
    query_dict,
    abstracts,
    data_ids,
    similarity_threshold,
    distance_threshold,
    top_n=5,
    model="mini",
):
    encoder = _load_model(model)
    encoded_abstracts = [_prepare_text(abstract, model, "passage") for abstract in abstracts]
    reference_embeddings = encoder.encode(encoded_abstracts)
    predictions = {}
    not_ranked = []

    for query_id, query_text in query_dict.items():
        query_embedding = encoder.encode(_prepare_text(query_text, model, "query"))
        similarities = encoder.similarity(query_embedding, reference_embeddings).tolist()[0]

        if max(similarities) < similarity_threshold:
            not_ranked.append(query_id)
            continue

        sorted_similarities = np.sort(similarities)[::-1]
        if len(sorted_similarities) > 1 and sorted_similarities[0] - sorted_similarities[1] < distance_threshold:
            not_ranked.append(query_id)
            continue

        top_indices = np.argsort(-np.array(similarities))[:top_n]
        predictions[query_id] = [data_ids[item] for item in top_indices]

    return predictions, not_ranked


def rank_candidates_by_semantic_similarity(
    query_dict,
    abstracts,
    data_ids,
    candidate_dict,
    similarity_threshold,
    distance_threshold,
    model="mini",
    top_n=5,
):
    encoder = _load_model(model)
    encoded_abstracts = [_prepare_text(abstract, model, "passage") for abstract in abstracts]
    reference_embeddings = encoder.encode(encoded_abstracts)
    data_id_to_index = {data_id: index for index, data_id in enumerate(data_ids)}
    predictions = {}
    similarity_scores = {}

    for query_id, candidates in candidate_dict.items():
        if query_id not in query_dict:
            continue

        candidate_indices = [data_id_to_index[candidate] for candidate in candidates if candidate in data_id_to_index]
        if not candidate_indices:
            continue

        query_embedding = encoder.encode(_prepare_text(query_dict[query_id], model, "query"))
        candidate_embeddings = reference_embeddings[candidate_indices]
        similarities = encoder.similarity(query_embedding, candidate_embeddings)[0].tolist()

        if max(similarities) < similarity_threshold:
            continue

        sorted_similarities = np.sort(similarities)[::-1]
        if len(sorted_similarities) > 1 and sorted_similarities[0] - sorted_similarities[1] < distance_threshold:
            continue

        ranked_indices = np.argsort(-np.array(similarities))[:top_n]
        predictions[query_id] = [candidates[index] for index in ranked_indices]
        similarity_scores[query_id] = [similarities[index] for index in ranked_indices]

    return predictions, similarity_scores
