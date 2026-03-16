import gzip
import pickle

import pandas as pd

from src.sentence_similarity import get_similarity_threshold, get_distance_threshold, rank_by_semantic_similarity
from src.utils import clean_data, prepare_text_inputs, calculate_mrr_at5

PATH_COLLECTION_DATA = 'data/subtask4b_collection_data.pkl'

df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
summaries_df = pd.read_csv('data/summaries.tsv', sep='\t', names=['id', 'summary'])
summaries_dict = dict(zip(summaries_df['id'], summaries_df['summary']))
df_collection['summary'] = df_collection['cord_uid'].map(summaries_dict)
data_dict = df_collection.set_index('cord_uid').to_dict('index')
with gzip.open("data/texts.pkl.gz", 'rb') as f:
    whole_papers_dict = {k: v for k, v in pickle.load(f).items()}
whole_papers_dict = clean_data(whole_papers_dict)

all_texts_with_summary = prepare_text_inputs(df_collection, use_summary=True)
all_texts_with_abstract = prepare_text_inputs(df_collection, use_summary=False)
all_data_ids = df_collection['cord_uid'].tolist()
id_to_abstract = {uid: text for uid, text in zip(all_data_ids, all_texts_with_abstract)}
id_to_summary = {uid: text for uid, text in zip(all_data_ids, all_texts_with_summary)}

whole_papers_dict.update({k: v for k, v in id_to_abstract.items() if k not in whole_papers_dict})

training_df = pd.read_csv('data/subtask4b_query_tweets_train.tsv', sep='\t')
dev_df = pd.read_csv('data/subtask4b_query_tweets_dev.tsv', sep='\t')

training_queries = training_df['tweet_text'].tolist()
training_query_ids = training_df['post_id'].tolist()
training_gold_ids = training_df['cord_uid'].tolist()
dev_queries = dev_df['tweet_text'].tolist()
dev_query_ids = dev_df['post_id'].tolist()
dev_gold_ids = dev_df['cord_uid'].tolist()

queries = training_queries
query_ids = training_query_ids
gold_ids = training_gold_ids
gold_dict = dict(zip(query_ids, gold_ids))
total_queries = len(queries)
final_predictions = {qid: [] for qid in query_ids}
unresolved_queries = set(query_ids)
queries_dict = {pid: text for pid, text in zip(query_ids, queries)}


# e5_sim_mean, e5_sim_std = get_similarity_threshold(queries, gold_ids, data_dict, column="summary", model='e5')
# e5_distance_mean, e5_distance_std = get_distance_threshold(queries, all_texts_with_summary, all_data_ids, gold_ids, model='e5')
# print(e5_sim_mean)
# print(e5_sim_std)
# print(e5_distance_mean)
# print(e5_distance_std)

e5_sim_mean = 0.8681767515604537
e5_sim_std = 0.03640567204079826
e5_distance_mean = 0.02950115931303749
e5_distance_std = 0.023578871251888643

# t5_sim_mean, t5_sim_std = get_similarity_threshold(queries, gold_ids, data_dict, column="summary", model='t5')
# t5_distance_mean, t5_distance_std = get_distance_threshold(queries, all_texts_with_summary, all_data_ids, gold_ids, model='t5')
# print(t5_sim_mean)
# print(t5_sim_std)
# print(t5_distance_mean)
# print(t5_distance_std)

test_df = pd.read_csv('data/subtask4b_query_tweets_dev.tsv', sep='\t')

test_queries = test_df['tweet_text'].tolist()
test_query_ids = test_df['post_id'].tolist()
queries = test_queries
query_ids = test_query_ids
total_queries = len(queries)
final_predictions = {qid: [] for qid in query_ids}
unresolved_queries = set(query_ids)
queries_dict = {pid: text for pid, text in zip(query_ids, queries)}

gold_ids = test_df['cord_uid'].tolist()
gold_dict = dict(zip(query_ids, gold_ids))


MODELS = {'mini': 'sentence-transformers/all-MiniLM-L6-v2'}#,
          #'specter': 'sentence-transformers/allenai-specter',
          #'scibert': 'allenai/scibert_scivocab_uncased',
          #'mpnet': 'all-mpnet-base-v2',
          #'bge': 'BAAI/bge-large-en-v1.5'}
          #"gte": 'thenlper/gte-large',
          #"e5": 'intfloat/e5-large-v2',
          #"t5": "sentence-transformers/sentence-t5-xxl"}

# for model in MODELS.keys():
#     print(model)
#     predictions, _ = rank_by_semantic_similarity(queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=0, distance_threshold=0, model=model, top_n=10)
#     print(calculate_mrr_at5(predictions, gold_dict))

# mini
# 0.5422142857142861
# Summaries

print("Summaries")
queries_summaries_df = pd.read_csv('data/dev_tweet_summaries.tsv', sep='\t', names=['id', 'summary'])
query_summaries_dict = dict(zip(queries_summaries_df['id'], queries_summaries_df['summary']))
current_queries_dict = {k: query_summaries_dict[k] for k in query_ids}
for model in MODELS.keys():
    print(model)
    predictions, _ = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=0, distance_threshold=0, model=model, top_n=10)
    print(calculate_mrr_at5(predictions, gold_dict))


