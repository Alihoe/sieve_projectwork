import re
from collections import defaultdict

from tqdm import tqdm
import pickle
import gzip
import pandas as pd
from functools import lru_cache

PATH_COLLECTION_DATA = '../data/subtask4b_collection_data.pkl'

REGEX_PATTERNS = {
    'fix_whitespace': [
        (re.compile(r"(\w)'s\b"), r"\1"),
        (re.compile(r"(\w's)'(\w)"), r"\1 '\2"),
        (re.compile(r"(?<=\w)'(?!s\b)(?=\w)"), " '"),
        (re.compile(r'(?<=[.,?!])(?=[^\s])'), ' '),
        (re.compile(r'(?<=\S)(?=[({\[])'), ' '),
        (re.compile(r'(?<=[)}\]])(?=\S)'), ' '),
        (re.compile(r'(?<=\S)-(?=\S)'), ' - '),
        (re.compile(r'(?<=\S)/(?=\S)'), ' / '),
        (re.compile(r'(?<=:)(?=\S)'), ' '),
        (re.compile(r'(?<=\S)&(?=\S)'), ' & '),
        (re.compile(r'\s+'), ' '),
    ],
    'quote_extraction': re.compile(r"(?<!\w's)\s*(['\"])(.*?)(?:\1|$)(?!\s*'s)"),
    'normalization1': re.compile(r'[“”‘’]'),
    'normalization2': re.compile(r'[^\w\s\'"-]'),
    'normalization3': re.compile(r'\s+'),
}


def fix_missing_whitespaces(texts):
    processed_texts = []
    for text in texts:
        for pattern, repl in REGEX_PATTERNS['fix_whitespace']:
            text = pattern.sub(repl, text)
        processed_texts.append(text.strip())
    return processed_texts


def remove_references(paper_dict):
    ref_patterns = [
        r'\n\s*references?\b.*\n',
        r'\n\s*bibliography\b.*\n',
        r'\n\s*works?\s*cited\b.*\n',
        r'\n\s*literature\s*cited\b.*\n',
        r'\n\s*citations?\b.*\n',
        r'\n\s*sources?\b.*\n',
        r'\n\s*acknowledgements?\b.*\n',
        r'\n\s*appendix\b.*\n',
        r'\n\s*r\s*e\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s\b.*\n',
    ]

    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in ref_patterns]

    processed_papers = {}

    for paper_id, text in paper_dict.items():
        earliest_match = None
        for pattern in compiled_patterns:
            match = pattern.search(text)
            if match and (earliest_match is None or match.start() < earliest_match.start()):
                earliest_match = match

        if earliest_match:
            processed_text = text[:earliest_match.start()].strip()
        else:
            processed_text = text

        processed_papers[paper_id] = processed_text

    return processed_papers


def extract_quotes(queries_dict):
    texts = list(queries_dict.values())
    ids = list(queries_dict.keys())
    if len(texts) != len(ids):
        raise ValueError("The number of texts must match the number of ids")
    fixed_texts = fix_missing_whitespaces(texts)
    result = {}
    for text, id_ in zip(fixed_texts, ids):
        quotes = REGEX_PATTERNS['quote_extraction'].findall(text)
        quote_contents = [content for _, content in quotes if len(content.split()) > 3]
        result[id_] = quote_contents
    return result


def clean_data(data_dict):
    cleaned = {}
    for key, value in data_dict.items():
        if value is None or (isinstance(value, float) and value != value):
            cleaned[key] = ""
        else:
            cleaned[key] = str(value)
    return cleaned


def rank_by_quotes(queries_dict, all_data_ids, data_dict_metadata, texts_file="../data/texts.pkl.gz"):
    with gzip.open(texts_file, 'rb') as f:
        data_dict = {k: v for k, v in pickle.load(f).items() if k in all_data_ids}
    data_dict = remove_references(clean_data(data_dict))
    query_dict = extract_quotes(queries_dict)
    predictions = {}
    not_ranked = []
    normalized_texts = {
        k: v.lower() if not pd.isna(v) else ""
        for k, v in data_dict.items()
    }

    for query_id, quotes in query_dict.items():
        candidates = []
        if not quotes:
            not_ranked.append(query_id)
            continue
        normalized_quotes = [q.lower() if q is not None else "" for q in quotes]
        for data_id, text in normalized_texts.items():
            for q in normalized_quotes:
                if q in text or q in data_dict_metadata[data_id]["title"].lower():
                    candidates.append(data_id)
        if candidates and len(candidates) < 4:
            predictions[query_id] = candidates
        else:
            not_ranked.append(query_id)
        #             quote_to_data_ids[q].append(data_id)
        # print(quote_to_data_ids)
        # unique_quote_data = []
        # for q in normalized_quotes:
        #     if len(quote_to_data_ids.get(q, [])) == 1:
        #         unique_quote_data.append(quote_to_data_ids[q][0])
        # if unique_quote_data:
        #     predictions[query_id] = unique_quote_data
        # else:
        #     not_ranked.append(query_id)
    return predictions, not_ranked


# texts = [
# "keep knocking them out of the field, aj. \"we further report a distorted t cell receptor repertoire in covid-19 patients with severe hyperinflammation, in support of such a superantigenic effect.\"",
#     "A covid'strain' emerged in spain in the summer and by mid-september, the strain was at 80% dominance in wales and scotland. this strain's'spread' was not attributed to 'infectiousness' but due to travel, lack of effective screening, and containment.",
#     "breaking: recent study reveals people without symptoms do not spread the virus!  \"there was no evidence of Transmission from asymptomatic positive persons to traced close contacts.\" time to lift restrictions?",
#     "Research on social media use and mental health in girls shows a correlation of r =.20, which many psychologists have traditionally viewed as \"small.\" However, this perception is shifting. Such effect sizes are \"the indispensable foundation for a cumulative psychological science."
#
# ]
# ids = [1, 2, 3, 4]
#
# queries_dict = dict(zip(ids, texts))
# df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
# all_data_ids = df_collection['cord_uid'].tolist()
# data_dict = df_collection.set_index('cord_uid').to_dict('index')
# predictions, not_ranked = rank_by_quotes(queries_dict, all_data_ids, data_dict)
# print(predictions)
# print(not_ranked)

# df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
# data_dict = df_collection.set_index('cord_uid').to_dict('index')
# training_df = pd.read_csv('../data/subtask4b_query_tweets_train.tsv', sep='\t')
# training_queries = training_df['tweet_text'].tolist()
# training_query_ids = training_df['post_id'].tolist()
# queries_dict = {pid: text for pid, text in zip(training_query_ids, training_queries)}
#
# predictions_qd, not_ranked_qd = rank_by_quotes(queries_dict, list(data_dict.keys()))


