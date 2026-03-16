import csv
import gzip
import pickle
import os
import pandas as pd
from src.author_matching import match_authors_to_queries
from src.bm252 import calculate_threshold, rank_all_queries
from src.journal_matching import rank_by_journal
from src.llm_re_ranking import rerank_queries_with_ollama
from src.quote_detection import rank_by_quotes
from src.sentence_similarity import rank_by_semantic_similarity, get_similarity_threshold, get_distance_threshold
from src.title_matching import rank_by_title
from src.token_matching import find_queries_with_seldom_tokens
from src.utils import clean_data, prepare_text_inputs, fill_dict2, save_query_results, truncate_dict, calculate_mrr_at5

PATH_COLLECTION_DATA = 'data/subtask4b_collection_data.pkl'

df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
summaries_df = pd.read_csv('data/summaries.tsv', sep='\t', names=['id', 'summary'])
summaries_dict = dict(zip(summaries_df['id'], summaries_df['summary']))
df_collection['summary'] = df_collection['cord_uid'].map(summaries_dict)
data_dict = df_collection.set_index('cord_uid').to_dict('index')
with gzip.open("data/texts.pkl.gz", 'rb') as f:
    whole_papers_dict = {k: v for k, v in pickle.load(f).items()}
whole_papers_dict = clean_data(whole_papers_dict)

queries_summaries_df = pd.read_csv('data/test_tweet_summaries.tsv', sep='\t', names=['id', 'summary'])
query_summaries_dict = dict(zip(queries_summaries_df['id'], queries_summaries_df['summary']))

all_texts_with_summary = prepare_text_inputs(df_collection, use_summary=True)
all_texts_with_abstract = prepare_text_inputs(df_collection, use_summary=False)
all_data_ids = df_collection['cord_uid'].tolist()
id_to_abstract = {uid: text for uid, text in zip(all_data_ids, all_texts_with_abstract)}
id_to_summary = {uid: text for uid, text in zip(all_data_ids, all_texts_with_summary)}

whole_papers_dict.update({k: v for k, v in id_to_abstract.items() if k not in whole_papers_dict})

#### CHANGE THIS ####
test_df = pd.read_csv('data/subtask4b_query_tweets_test.tsv', sep='\t')
#####################

test_queries = test_df['tweet_text'].tolist()
test_query_ids = test_df['post_id'].tolist()
queries = test_queries
query_ids = test_query_ids
total_queries = len(queries)
final_predictions = {qid: [] for qid in query_ids}
unresolved_queries = set(query_ids)
queries_dict = {pid: text for pid, text in zip(query_ids, queries)}


mean_similarity = 0.6517946525885147
std_similarity = 0.15747238809867273
mean_distance = 0.10337032085336031
std_distance = 0.09395258399486513
similarity_threshold = mean_similarity
distance_threshold = mean_distance
n_candidates = 30
generous_similarity_threshold = mean_similarity - std_similarity
generous_distance_threshold = mean_distance - std_distance

mean_similarity_sci = 0.8608020880380364
std_similarity_sci = 0.05164753269378038
mean_distance_sci = 0.01344577842134081
std_distance_sci = 0.012691256058858872
similarity_threshold_sci = mean_similarity_sci + std_similarity_sci
distance_threshold_sci = mean_distance_sci + std_distance_sci

mean_similarity_mpnet=0.6848235173594495
std_similarity_mpnet=0.1605105865157184
mean_distance_mpnet=0.09782780249794855
std_distance_mpnet=0.1605105865157184
similarity_threshold_mpnet = mean_similarity_mpnet
distance_threshold_mpnet = mean_distance_mpnet
generous_similarity_threshold_mpnet = mean_similarity_mpnet - std_similarity_mpnet
generous_distance_threshold_mpnet = mean_distance_mpnet - std_distance_mpnet

mean_similarity_spc = 0.8423419033753903
std_similarity_spc = 0.07132700116476645
mean_distance_spc = 0.041100625714210616
std_distance_spc = 0.03623312174309394
similarity_threshold_spc = mean_similarity_spc
distance_threshold_spc = mean_distance_spc

bm25_abstracts_mean = 61.071572593218015
bm25_abstracts_std = 31.905156653723697
bm25_abstracts_threshold = bm25_abstracts_mean
bm25_whole_papers_mean = 66.90418735869673
bm25_whole_papers_std = 43.744064315317125
bm25_whole_papers_threshold = bm25_whole_papers_mean + bm25_abstracts_threshold

gte_sim_mean = 0.8836162919980732
gte_sim_std = 0.04797421730970624
gte_distance_mean = 0.037110097138704765
gte_distance_std = 0.029861044781690313

similarity_threshold_gte = gte_sim_mean + gte_sim_std
distance_threshold_gte = gte_distance_mean + gte_distance_std

bge_sim_mean = 0.7730291020236738
bge_sim_std = 0.09325022222541414
bge_distance_mean = 0.06336793244908033
bge_distance_std = 0.0509909760582547

similarity_threshold_bge = bge_sim_mean + bge_sim_std
distance_threshold_bge = bge_distance_mean + bge_distance_std

e5_sim_mean = 0.8681767515604537
e5_sim_std = 0.03640567204079826
e5_distance_mean = 0.02950115931303749
e5_distance_std = 0.023578871251888643

similarity_threshold_e5 = e5_sim_mean + e5_sim_std
distance_threshold_e5 = e5_distance_mean + e5_distance_std


BEST_MODEL = "bge"

print(f"Starting pipeline with {len(queries)} queries")

final_predictions = {}

print("\n Component: Title Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
predictions_td, not_ranked_td = rank_by_title(current_queries_dict, data_dict)
predictions_qd = rerank_queries_with_ollama(predictions_td, queries_dict, data_dict, max_candidates=n_candidates)
final_predictions= final_predictions | predictions_td
unresolved_queries &= set(not_ranked_td)
resolved_now = set(current_queries) - set(not_ranked_td)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))

print("\n Component: Quote Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
predictions_qd, not_ranked_qd = rank_by_quotes(current_queries_dict, all_data_ids, data_dict, texts_file="data/texts.pkl.gz")
predictions_qd = rerank_queries_with_ollama(predictions_qd, queries_dict, data_dict, max_candidates=n_candidates)
final_predictions = final_predictions | predictions_qd
unresolved_queries &= set(not_ranked_qd)
resolved_now = set(current_queries) - set(not_ranked_qd)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))


print("\n Component: BM25 on Abstracts")
current_queries = list(unresolved_queries)
current_queries_dict = {k: query_summaries_dict[k] for k in current_queries}
predictions_bm25, not_ranked_bm25 = rank_all_queries(current_queries_dict, id_to_abstract, threshold=bm25_abstracts_threshold, top_k=n_candidates)
final_predictions = final_predictions | predictions_bm25
unresolved_queries &= set(not_ranked_bm25)
resolved_now = set(current_queries) - set(not_ranked_bm25)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))


print("\n Component: Similarity MPNET Small")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
predictions_sus, not_ranked_sus = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold, distance_threshold=distance_threshold, top_n=n_candidates)
final_predictions = final_predictions | predictions_sus
unresolved_queries &= set(not_ranked_sus)
resolved_now = set(current_queries) - set(not_ranked_sus)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))



print("\n Component: Similarity MPNET")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
predictions_mpnet, not_ranked_mpnet = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold_mpnet, distance_threshold=distance_threshold_mpnet, top_n=n_candidates, model="mpnet")
final_predictions = final_predictions | predictions_mpnet
unresolved_queries &= set(not_ranked_mpnet)
resolved_now = set(current_queries) - set(not_ranked_mpnet)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))


print("\n Component: Similarity SPECTER")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
predictions_specter, not_ranked_specter = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold_spc, distance_threshold=distance_threshold_spc, top_n=n_candidates, model="specter")
final_predictions = final_predictions | predictions_specter
unresolved_queries &= set(not_ranked_specter)
resolved_now = set(current_queries) - set(not_ranked_specter)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))


print("\n Component: Unique Token Linking")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
predictions_untr, not_ranked_untr = find_queries_with_seldom_tokens(current_queries_dict, id_to_abstract)
final_predictions = final_predictions | predictions_untr
unresolved_queries &= set(not_ranked_untr)
resolved_now = set(current_queries) - set(not_ranked_untr)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))

print("\n Component: Journal Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
predictions_jd, not_ranked_jd = rank_by_journal(current_queries_dict, data_dict)
predictions_jd = rerank_queries_with_ollama(predictions_jd, queries_dict, data_dict, max_candidates=n_candidates)
final_predictions = final_predictions | predictions_jd
unresolved_queries &= set(not_ranked_jd)
resolved_now = set(current_queries) - set(not_ranked_jd)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))

print("\n Component: Author Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
predictions_ad, not_ranked_ad = match_authors_to_queries(data_dict, current_queries_dict)
predictions_ad = rerank_queries_with_ollama(predictions_ad, queries_dict, data_dict, max_candidates=n_candidates)
final_predictions.update(predictions_ad)
unresolved_queries &= set(not_ranked_ad)
resolved_now = set(current_queries) - set(not_ranked_ad)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))


print("\n Component: Final Summary Similarity No Thresholds")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
predictions_bge_final_test, not_ranked_bge_final_test = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=0, distance_threshold=0, model=BEST_MODEL, top_n=n_candidates)
final_predictions = final_predictions | predictions_bge_final_test
unresolved_queries &= set(not_ranked_bge_final_test)
resolved_now = set(current_queries) - set(not_ranked_bge_final_test)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))

print("\n Component: Fill with Similarity No Thresholds")
all_predictions_best, _ = rank_by_semantic_similarity(queries_dict, all_texts_with_summary,
                                                                        all_data_ids, similarity_threshold=0,
                                                                        distance_threshold=0, model=BEST_MODEL,
                                                                        top_n=n_candidates)
final_predictions = fill_dict2(all_predictions_best, final_predictions, n_candidates)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("Length of Final predictions: "+ str(len(final_predictions)))

final_predictions = truncate_dict(final_predictions)


with open('TEST_PREDICTIONS_ORIGINAL.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(["post_id", "preds"])
    for key, value in final_predictions.items():
        writer.writerow([key, value])
