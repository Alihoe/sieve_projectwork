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


def extract_tokens(texts):
    nlp = spacy.load("en_core_web_sm")
    results = []
    for text in texts:
        doc = nlp(text)
        # Extract nouns and proper nouns as important tokens
        tokens = [token.text.lower() for token in doc
                 if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop]
        results.append(tokens)
    return results


def precompute_data_tokens(data_texts, cache_file="data/data_tokens.pkl"):
    cache_path = Path(cache_file)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    data_tokens = {}
    for data_id, text in data_texts.items():
        data_tokens[data_id] = extract_tokens([text])[0]

    with open(cache_path, 'wb') as f:
        pickle.dump(data_tokens, f)
    return data_tokens


def find_queries_with_seldom_tokens(query_texts, data_texts, threshold=5, min_shared_tokens=2):
    data_tokens_cache = precompute_data_tokens(data_texts)

    global_token_counts = defaultdict(int)
    for tokens in data_tokens_cache.values():
        for token in tokens:
            global_token_counts[token] += 1

    ranked = {}
    not_ranked = []

    for query_id, text in query_texts.items():
        query_tokens = set(extract_tokens([text])[0])

        if not query_tokens:
            not_ranked.append(query_id)
            continue

        seldom_tokens = {token for token in query_tokens if global_token_counts.get(token, 0) <= threshold}

        if not seldom_tokens:
            not_ranked.append(query_id)
            continue

        matching_data = []
        for data_id, data_tokens in data_tokens_cache.items():
            common_tokens = set(data_tokens) & seldom_tokens
            if len(common_tokens) >= min_shared_tokens:
                matching_data.append((data_id, len(common_tokens), common_tokens))

        if matching_data:
            # print(f"\nProcessing query {query_id}: '{text}'")
            # print(f"Found {len(matching_data)} documents sharing ≥{min_shared_tokens} seldom tokens")
            matching_data.sort(key=lambda x: -x[1])
            ranked[query_id] = [data_id for data_id, _, _ in matching_data]
            #
            # for i, (data_id, count, common_tokens) in enumerate(matching_data[:3]):
            #     print(f"Match #{i + 1} (shared {count} tokens):")
            #     print(f"  Document ID: {data_id}")
            #     print(f"  Common tokens: {', '.join(common_tokens)}")
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
# find_queries_with_seldom_tokens(queries_dict, id_to_abstract, threshold=3)


