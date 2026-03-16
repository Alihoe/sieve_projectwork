import csv
import gzip
import pickle

import pandas as pd
from ollama import generate

SUMMARIZATION_MODEL = "llama3.1:8b"


def summarize(texts, ids):
    summaries = []
    for idx, text in enumerate(texts):
        prompt = "Summarize the following abstract into three sentences. Only output the summary without any explanation or introduction. The summary should start with \"The paper contains\":\n"+text
        summary = generate(model=SUMMARIZATION_MODEL, prompt=prompt)['response']
        new_row = [ids[idx], summary]
        with open('summaries.tsv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(new_row)
        summaries.append(summary)
    return summaries


def summarize_to_sentence(texts, ids):
    summaries = []
    for idx, text in enumerate(texts):
        prompt = "Summarize the following abstract into one sentence. Only output the summary without any explanation or introduction. The summary should start with \"The paper contains\":\n"+text
        summary = generate(model=SUMMARIZATION_MODEL, prompt=prompt)['response']
        new_row = [ids[idx], summary]
        with open('one_sentence_summaries.tsv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(new_row)
        summaries.append(summary)
    return summaries


def summarize_papers(texts, ids):
    summaries = []
    for idx, text in enumerate(texts):
        print(idx)
        prompt = "Summarize the following paper. Only output the summary without any explanation or introduction or conclusion or other follow-up text. Do not ask what you should do next. The summary should consist of approximately five whole sentences. The summary should start with \"The paper contains\":\n"+text
        summary = generate(model=SUMMARIZATION_MODEL, prompt=prompt)['response']
        new_row = [ids[idx], summary]
        with open('../data/paper_summaries.tsv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(new_row)
        summaries.append(summary)
    return summaries


def summarize_tweets(query_dict, file_path='../data/test_tweet_summaries.tsv'):
    summary_dict = {}
    for key, value in query_dict.items():
        prompt= "The following tweet refers to a scientific paper. Summarize the content of the paper in one sentence. If you do not find a reference to a scientific paper, justreformulate the tweet in a precise sentence. Return ONLY this sentence, no explanations.\n"+value
        summary = generate(model=SUMMARIZATION_MODEL, prompt=prompt)['response']
        summary_dict[key] = summary
        new_row = [key, summary]
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(new_row)
    return summary_dict

# texts_file="../data/texts.pkl.gz"
# with gzip.open(texts_file, 'rb') as f:
#     id_text_dict = pickle.load(f)
# texts = list(id_text_dict.values())
# ids = list(id_text_dict.keys())
# summarize_papers(texts, ids)

test_df = pd.read_csv('../data/subtask4b_query_tweets_test.tsv', sep='\t')
test_queries = test_df['tweet_text'].tolist()
test_query_ids = test_df['post_id'].tolist()
queries = test_queries
query_ids = test_query_ids
total_queries = len(queries)
final_predictions = {qid: [] for qid in query_ids}
unresolved_queries = set(query_ids)
queries_dict = {pid: text for pid, text in zip(query_ids, queries)}
summarize_tweets(queries_dict)

test_df = pd.read_csv('../data/subtask4b_query_tweets_dev.tsv', sep='\t')

test_queries = test_df['tweet_text'].tolist()
test_query_ids = test_df['post_id'].tolist()
queries = test_queries
query_ids = test_query_ids
total_queries = len(queries)
final_predictions = {qid: [] for qid in query_ids}
unresolved_queries = set(query_ids)
queries_dict = {pid: text for pid, text in zip(query_ids, queries)}
summarize_tweets(queries_dict, file_path='../data/dev_tweet_summaries.tsv')
