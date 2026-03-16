import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy


def prepare_text_inputs(df, use_summary=False):
    texts = []
    for _, row in df.iterrows():
        title = str(row['title'])
        summary = str(row.get('summary', ''))
        abstract = str(row.get('abstract', ''))
        if use_summary and summary and len(abstract) > 100:
            text = f"{title}: {summary}"
        elif not use_summary and len(abstract) > 100:
            text = f"{title}: {abstract}"
        else:
            text = title
        texts.append(text.strip())
    return texts


def extract_wikidata_ids(texts):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    results = []
    for text in texts:
        doc = nlp(text)
        ids = [ent.get_id() for ent in doc._.linkedEntities]
        results.append(ids)
    return results


def precompute_data_nes(data_texts, cache_file="data/data_nes.pkl"):
    cache_path = Path(cache_file)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    data_nes = {}
    for data_id, text in data_texts.items():
        data_nes[data_id] = extract_wikidata_ids([text])[0]

    with open(cache_path, 'wb') as f:
        pickle.dump(data_nes, f)
    return data_nes
#
#
# def find_queries_with_seldom_nes_candidates(query_to_candidates, query_texts, data_texts, threshold=3):
#     data_nes_cache = precompute_data_nes(data_texts)
#     ranked = {}
#     not_ranked = []
#     for query_id, candidate_ids in query_to_candidates.items():
#         query_nes = set(extract_wikidata_ids([query_texts[query_id]])[0])
#         if not query_nes:
#             not_ranked.append(query_id)
#             continue
#         ne_counts = defaultdict(int)
#         for data_id in candidate_ids:
#             for ne in data_nes_cache.get(data_id, []):
#                 ne_counts[ne] += 1
#         seldom_nes = {ne for ne in query_nes if ne_counts.get(ne, 0) <= threshold}
#         if not seldom_nes:
#             not_ranked.append(query_id)
#             continue
#         matching_data = []
#         for data_id in candidate_ids:
#             common_nes = set(data_nes_cache.get(data_id, [])) & seldom_nes
#             if common_nes:
#                 matching_data.append((data_id, len(common_nes)))
#         if matching_data:
#             matching_data.sort(key=lambda x: -x[1])
#             ranked[query_id] = [data_id for data_id, count in matching_data]
#         else:
#             not_ranked.append(query_id)
#     return ranked, not_ranked


def find_queries_with_seldom_nes(query_texts, data_texts, threshold=3, min_shared_nes=2):
    data_nes_cache = precompute_data_nes(data_texts)

    global_ne_counts = defaultdict(int)
    for nes in data_nes_cache.values():
        for ne in nes:
            global_ne_counts[ne] += 1

    ranked = {}
    not_ranked = []

    for query_id, text in query_texts.items():
        query_nes = set(extract_wikidata_ids([text])[0])

        if not query_nes:
            not_ranked.append(query_id)
            continue

        seldom_nes = {ne for ne in query_nes if global_ne_counts.get(ne, 0) <= threshold}

        if not seldom_nes:
            not_ranked.append(query_id)
            continue

        matching_data = []
        for data_id, data_nes in data_nes_cache.items():
            common_nes = set(data_nes) & seldom_nes
            if len(common_nes) >= min_shared_nes:  # Only count if ≥2 shared seldom NEs
                matching_data.append((data_id, len(common_nes), common_nes))

        if matching_data:
            # print(f"\nProcessing query {query_id}: '{text}'")
            # print(f"Found {len(matching_data)} documents sharing ≥{min_shared_nes} seldom NEs")
            matching_data.sort(key=lambda x: -x[1])
            ranked[query_id] = [data_id for data_id, _, _ in matching_data]

            # for i, (data_id, count, common_nes) in enumerate(matching_data[:3]):
            #     print(f"Match #{i + 1} (shared {count} NEs):")
            #     print(f"  Document ID: {data_id}")
            #     print(data_texts[data_id])
        else:
            not_ranked.append(query_id)

    return ranked, not_ranked


# PATH_COLLECTION_DATA = '../data/subtask4b_collection_data.pkl'
# df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
# all_texts_with_abstract = prepare_text_inputs(df_collection, use_summary=False)
# all_data_ids = df_collection['cord_uid'].tolist()
# id_to_abstract = {uid: text for uid, text in zip(all_data_ids, all_texts_with_abstract)}
# training_df = pd.read_csv('../data/subtask4b_query_tweets_train.tsv', sep='\t')
# training_queries = training_df['tweet_text'].tolist()
# training_query_ids = training_df['post_id'].tolist()
# queries_dict = {pid: text for pid, text in zip(training_query_ids, training_queries)}
# find_queries_with_seldom_nes(queries_dict, id_to_abstract, threshold=3)