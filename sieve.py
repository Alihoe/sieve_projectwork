import csv
import gzip
import pickle
import os
import pandas as pd
from src.author_matching import match_authors_to_queries
from src.bm252 import calculate_threshold, rank_all_queries
from src.journal_matching import rank_by_journal
from src.llm_re_ranking import rerank_queries_with_ollama
from src.named_entity_ranking import find_queries_with_seldom_nes
from src.quote_detection import rank_by_quotes
from src.sentence_similarity import rank_by_semantic_similarity, get_similarity_threshold, get_distance_threshold
from src.title_matching import rank_by_title
from src.token_matching import find_queries_with_seldom_tokens
from src.utils import clean_data, prepare_text_inputs, calculate_mrr, calculate_recall, fill_dict2, save_query_results

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

get_similarity_threshold, get_distance_threshold

#gte_sim_mean, gte_sim_std = get_similarity_threshold(queries, gold_ids, data_dict, column="summary", model='gte')
#gte_distance_mean, gte_distance_std = get_distance_threshold(queries, all_texts_with_summary, all_data_ids, gold_ids, model='gte')
# print(gte_sim_mean)
# print(gte_sim_std)
# print(gte_distance_mean)
# print(gte_distance_std)

bge_sim_mean, bge_sim_std = get_similarity_threshold(queries, gold_ids, data_dict, column="summary", model='bge')
bge_distance_mean, bge_distance_std = get_distance_threshold(queries, all_texts_with_summary, all_data_ids, gold_ids, model='bge')
print(bge_sim_mean)
print(bge_sim_std)
print(bge_distance_mean)
print(bge_distance_std)

bge_sim_mean = 0.7730291020236738
bge_sim_std = 0.09325022222541414
bge_distance_mean = 0.06336793244908033
bge_distance_std = 0.0509909760582547


# gte_sim_mean = 0.8836162919980732
# gte_sim_std = 0.04797421730970624
# gte_distance_mean = 0.037110097138704765
# gte_distance_std = 0.029861044781690313


mean_similarity = 0.6517946525885147
std_similarity = 0.15747238809867273
mean_distance = 0.10337032085336031
std_distance = 0.09395258399486513
similarity_threshold = mean_similarity #+ std_similarity
distance_threshold = mean_distance #+ std_distance
n_candidates = 30
generous_similarity_threshold = mean_similarity #- std_similarity
generous_distance_threshold = mean_distance #- std_distance

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

print(f"Starting pipeline with {len(queries)} queries")

final_predictions = {}

print("\n Component: Title Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/title_predictions.pkl') and os.path.exists('data/component_predictions/title_not_ranked.pkl'):
    with open('data/component_predictions/title_predictions.pkl', 'rb') as f:
        predictions_td = pickle.load(f)
    with open('data/component_predictions/title_not_ranked.pkl', 'rb') as f:
        not_ranked_td = pickle.load(f)
else:
    predictions_td, not_ranked_td = rank_by_title(current_queries_dict, data_dict)
    with open('data/component_predictions/title_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_td, f)
    with open('data/component_predictions/title_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_td, f)
final_predictions= final_predictions | predictions_td
unresolved_queries &= set(not_ranked_td)
resolved_now = set(current_queries) - set(not_ranked_td)
print(f"MRR for resolved: {calculate_mrr(predictions_td, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_td, gold_dict, data_dict, queries_dict, "no_recall/no_recall_td.txt")))

print("\n Component: Quote Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/quote_predictions.pkl') and os.path.exists('data/component_predictions/quote_not_ranked.pkl'):
    with open('data/component_predictions/quote_predictions.pkl', 'rb') as f:
        predictions_qd = pickle.load(f)
    with open('data/component_predictions/quote_not_ranked.pkl', 'rb') as f:
        not_ranked_qd = pickle.load(f)
else:
    predictions_qd, not_ranked_qd = rank_by_quotes(current_queries_dict, all_data_ids, data_dict, texts_file="data/texts.pkl.gz")
    predictions_qd = rerank_queries_with_ollama(predictions_qd, queries_dict, data_dict, max_candidates=n_candidates)
    with open('data/component_predictions/quote_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_qd, f)
    with open('data/component_predictions/quote_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_qd, f)
final_predictions= final_predictions | predictions_qd
unresolved_queries &= set(not_ranked_qd)
resolved_now = set(current_queries) - set(not_ranked_qd)
print(f"MRR for resolved: {calculate_mrr(predictions_qd, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_qd, gold_dict, data_dict, queries_dict, "no_recall/no_recall_qd.txt")))

print("\n Component: Summary Similarity")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/summary_predictions.pkl') and os.path.exists('data/component_predictions/summary_not_ranked.pkl'):
    with open('data/component_predictions/summary_predictions.pkl', 'rb') as f:
        predictions_sus = pickle.load(f)
    with open('data/component_predictions/summary_not_ranked.pkl', 'rb') as f:
        not_ranked_sus = pickle.load(f)
else:
    predictions_sus, not_ranked_sus = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold, distance_threshold=distance_threshold, top_n=n_candidates)
    with open('data/component_predictions/summary_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_sus, f)
    with open('data/component_predictions/summary_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_sus, f)
final_predictions = final_predictions | predictions_sus
unresolved_queries &= set(not_ranked_sus)
resolved_now = set(current_queries) - set(not_ranked_sus)
print(f"MRR for resolved: {calculate_mrr(predictions_sus, gold_dict)}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("RECALL: "+str(calculate_recall(predictions_sus, gold_dict, data_dict, queries_dict, "no_recall/no_recall_sus.txt")))

print("\n Component: BM25 on Abstracts")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/bm25_abstracts_predictions.pkl') and os.path.exists('data/component_predictions/bm25_abstracts_not_ranked.pkl'):
    with open('data/component_predictions/bm25_abstracts_predictions.pkl', 'rb') as f:
        predictions_bm25 = pickle.load(f)
    with open('data/component_predictions/bm25_abstracts_not_ranked.pkl', 'rb') as f:
        not_ranked_bm25 = pickle.load(f)
else:
    predictions_bm25, not_ranked_bm25 = rank_all_queries(current_queries_dict, id_to_abstract, threshold=bm25_abstracts_threshold, top_k=n_candidates)
    with open('data/component_predictions/bm25_abstracts_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_bm25, f)
    with open('data/component_predictions/bm25_abstracts_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_bm25, f)
overlap = set(predictions_sus.keys()) & set(predictions_bm25.keys())
final_predictions = final_predictions | predictions_bm25
unresolved_queries &= set(not_ranked_bm25)
resolved_now = set(current_queries) - set(not_ranked_bm25)
print(f"MRR for resolved: {calculate_mrr(predictions_bm25, gold_dict)}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print("RECALL: "+str(calculate_recall(predictions_bm25, gold_dict, data_dict, queries_dict, "no_recall/no_recall_bm25.txt")))

print("\n Component: Scientific Similarity SPECTER")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/specter_predictions.pkl') and os.path.exists('data/component_predictions/specter_not_ranked.pkl'):
    with open('data/component_predictions/specter_predictions.pkl', 'rb') as f:
        predictions_specs = pickle.load(f)
    with open('data/component_predictions/specter_not_ranked.pkl', 'rb') as f:
        not_ranked_specs = pickle.load(f)
else:
    predictions_specs, not_ranked_specs = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold_spc, distance_threshold=distance_threshold_spc, model="specter", top_n=n_candidates)
    with open('data/component_predictions/specter_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_specs, f)
    with open('data/component_predictions/specter_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_specs, f)
final_predictions= final_predictions | predictions_specs
unresolved_queries &= set(not_ranked_specs)
resolved_now = set(current_queries) - set(not_ranked_specs)
print(f"MRR for resolved: {calculate_mrr(predictions_specs, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_specs, gold_dict, data_dict, queries_dict, "no_recall/no_recall_specs.txt")))

print("\n Component: Summary Similarity MPNET")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/mpnet_predictions.pkl') and os.path.exists('data/component_predictions/mpnet_not_ranked.pkl'):
    with open('data/component_predictions/mpnet_predictions.pkl', 'rb') as f:
        predictions_mpnet = pickle.load(f)
    with open('data/component_predictions/mpnet_not_ranked.pkl', 'rb') as f:
        not_ranked_mpnet = pickle.load(f)
else:
    predictions_mpnet, not_ranked_mpnet = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold_mpnet, distance_threshold=distance_threshold_mpnet, model="mpnet", top_n=n_candidates)
    with open('data/component_predictions/mpnet_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_mpnet, f)
    with open('data/component_predictions/mpnet_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_mpnet, f)
final_predictions = final_predictions | predictions_mpnet
unresolved_queries &= set(not_ranked_mpnet)
resolved_now = set(current_queries) - set(not_ranked_mpnet)
print(f"MRR for resolved: {calculate_mrr(predictions_mpnet, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_mpnet, gold_dict, data_dict, queries_dict, "no_recall/no_recall_mpnet.txt")))

print("\n Component: Scientific Similarity")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/scibert_predictions.pkl') and os.path.exists('data/component_predictions/scibert_not_ranked.pkl'):
    with open('data/component_predictions/scibert_predictions.pkl', 'rb') as f:
        predictions_scis = pickle.load(f)
    with open('data/component_predictions/scibert_not_ranked.pkl', 'rb') as f:
        not_ranked_scis = pickle.load(f)
else:
    predictions_scis, not_ranked_scis = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold_sci, distance_threshold=distance_threshold_sci, model="scibert", top_n=n_candidates)
    with open('data/component_predictions/scibert_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_scis, f)
    with open('data/component_predictions/scibert_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_scis, f)
final_predictions= final_predictions | predictions_scis
unresolved_queries &= set(not_ranked_scis)
resolved_now = set(current_queries) - set(not_ranked_scis)
print(f"MRR for resolved: {calculate_mrr(predictions_scis, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_scis, gold_dict, data_dict, queries_dict, "no_recall/no_recall_scis.txt")))

# print("\n Component: BM25 on Whole Papers")
# current_queries = list(unresolved_queries)
# current_queries_dict = {id: queries_dict[id] for id in current_queries if id in queries_dict}
# if os.path.exists('data/component_predictions/bm25_whole_predictions.pkl') and os.path.exists('data/component_predictions/bm25_whole_not_ranked.pkl'):
#     with open('data/component_predictions/bm25_whole_predictions.pkl', 'rb') as f:
#         predictions_bm25wp = pickle.load(f)
#     with open('data/component_predictions/bm25_whole_not_ranked.pkl', 'rb') as f:
#         not_ranked_bm25wp = pickle.load(f)
# else:
#     predictions_bm25wp, not_ranked_bm25wp = rank_all_queries(current_queries_dict, whole_papers_dict, threshold=bm25_whole_papers_threshold, top_k=n_candidates)
#     with open('data/component_predictions/bm25_whole_predictions.pkl', 'wb') as f:
#         pickle.dump(predictions_bm25wp, f)
#     with open('data/component_predictions/bm25_whole_not_ranked.pkl', 'wb') as f:
#         pickle.dump(not_ranked_bm25wp, f)
# unresolved_queries &= set(not_ranked_bm25wp)
# resolved_now = set(current_queries) - set(not_ranked_bm25wp)
# for qid in resolved_now:
#     final_predictions[qid] = predictions_bm25wp.get(qid, [])
# print(f"MRR for resolved: {calculate_mrr(predictions_bm25wp, gold_dict)}")
# mrr = calculate_mrr(final_predictions, gold_dict)
# print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
# print(f"MRR@1: {mrr[1]:.4f}, MRR@5: {mrr[5]:.4f}, MRR@10: {mrr[10]:.4f}")
# print("RECALL: "+str(calculate_recall(predictions_bm25wp, gold_dict, data_dict, queries_dict, "no_recall/no_recall_bm25wp.txt")))

print("\n Component: Unique Named Entity Linking")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/ne_linking_predictions.pkl') and os.path.exists('data/component_predictions/ne_linking_not_ranked.pkl'):
    with open('data/component_predictions/ne_linking_predictions.pkl', 'rb') as f:
        predictions_untr = pickle.load(f)
    with open('data/component_predictions/ne_linking_not_ranked.pkl', 'rb') as f:
        not_ranked_untr = pickle.load(f)
else:
    predictions_untr, not_ranked_untr = find_queries_with_seldom_nes(current_queries_dict, id_to_abstract)
    with open('data/component_predictions/ne_linking_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_untr, f)
    with open('data/component_predictions/ne_linking_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_untr, f)
final_predictions = final_predictions | predictions_untr
unresolved_queries &= set(not_ranked_untr)
resolved_now = set(current_queries) - set(not_ranked_untr)
print(f"MRR for resolved: {calculate_mrr(predictions_untr, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_untr, gold_dict, data_dict, queries_dict, "no_recall/no_recall_untr.txt")))

print("\n Component: Unique Token Linking")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/token_linking_predictions.pkl') and os.path.exists('data/component_predictions/token_linking_not_ranked.pkl'):
    with open('data/component_predictions/token_linking_predictions.pkl', 'rb') as f:
        predictions_untr = pickle.load(f)
    with open('data/component_predictions/token_linking_not_ranked.pkl', 'rb') as f:
        not_ranked_untr = pickle.load(f)
else:
    predictions_untr, not_ranked_untr = find_queries_with_seldom_tokens(current_queries_dict, id_to_abstract)
    with open('data/component_predictions/token_linking_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_untr, f)
    with open('data/component_predictions/token_linking_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_untr, f)
final_predictions = final_predictions | predictions_untr
unresolved_queries &= set(not_ranked_untr)
resolved_now = set(current_queries) - set(not_ranked_untr)
print(f"MRR for resolved: {calculate_mrr(predictions_untr, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_untr, gold_dict, data_dict, queries_dict, "no_recall/no_recall_untr.txt")))

print("\n Component: Journal Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/journal_predictions.pkl') and os.path.exists('data/component_predictions/journal_not_ranked.pkl'):
    with open('data/component_predictions/journal_predictions.pkl', 'rb') as f:
        predictions_jd = pickle.load(f)
    with open('data/component_predictions/journal_not_ranked.pkl', 'rb') as f:
        not_ranked_jd = pickle.load(f)
else:
    predictions_jd, not_ranked_jd = rank_by_journal(current_queries_dict, data_dict)
    #predictions_jd = rerank_queries_with_ollama(predictions_jd, queries_dict, data_dict, max_candidates=n_candidates)
    with open('data/component_predictions/journal_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_jd, f)
    with open('data/component_predictions/journal_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_jd, f)
final_predictions = final_predictions | predictions_jd
unresolved_queries &= set(not_ranked_jd)
resolved_now = set(current_queries) - set(not_ranked_jd)
print(f"MRR for resolved: {calculate_mrr(predictions_jd, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_jd, gold_dict, data_dict, queries_dict, "no_recall/no_recall_jd.txt")))

print("\n Component: Author Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/author_predictions.pkl') and os.path.exists('data/component_predictions/author_not_ranked.pkl'):
    with open('data/component_predictions/author_predictions.pkl', 'rb') as f:
        predictions_ad = pickle.load(f)
    with open('data/component_predictions/author_not_ranked.pkl', 'rb') as f:
        not_ranked_ad = pickle.load(f)
else:
    predictions_ad, not_ranked_ad = match_authors_to_queries(data_dict, current_queries_dict)
    #predictions_ad = rerank_queries_with_ollama(predictions_ad, queries_dict, data_dict, max_candidates=n_candidates)
    with open('data/component_predictions/author_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_ad, f)
    with open('data/component_predictions/author_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_ad, f)
final_predictions.update(predictions_ad)
unresolved_queries &= set(not_ranked_ad)
resolved_now = set(current_queries) - set(not_ranked_ad)
print(f"MRR for resolved: {calculate_mrr(predictions_ad, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_ad, gold_dict, data_dict, queries_dict, "no_recall/no_recall_ad.txt")))


print("\n Component: Summary Similarity MPNET Generous Thresholds")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/mpnet_gt_predictions.pkl') and os.path.exists('data/component_predictions/mpnet_gt_not_ranked.pkl'):
    with open('data/component_predictions/mpnet_gt_predictions.pkl', 'rb') as f:
        predictions_mpnet_gt = pickle.load(f)
    with open('data/component_predictions/mpnet_gt_not_ranked.pkl', 'rb') as f:
        not_ranked_mpnet_gt = pickle.load(f)
else:
    predictions_mpnet_gt, not_ranked_mpnet_gt = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=generous_similarity_threshold_mpnet, distance_threshold=generous_distance_threshold_mpnet, model="mpnet", top_n=n_candidates)
    with open('data/component_predictions/mpnet_gt_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_mpnet_gt, f)
    with open('data/component_predictions/mpnet_gt_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_mpnet_gt, f)
final_predictions = final_predictions | predictions_mpnet_gt
unresolved_queries &= set(not_ranked_mpnet_gt)
resolved_now = set(current_queries) - set(not_ranked_mpnet_gt)
print(f"MRR for resolved: {calculate_mrr(predictions_mpnet_gt, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_mpnet_gt, gold_dict, data_dict, queries_dict, "no_recall/no_recall_mpnet_gt.txt")))

print("\n Component: Fill with Summary Similarity MPNET No Thresholds")
all_predictions_mpnet_ntf, _ = rank_by_semantic_similarity(queries_dict, all_texts_with_summary,
                                                                        all_data_ids, similarity_threshold=0,
                                                                        distance_threshold=1, model="mpnet_nt",
                                                                        top_n=n_candidates)

final_predictions = fill_dict2(all_predictions_mpnet_ntf, final_predictions)


print("\n Component: Final Summary Similarity MPNET No Thresholds")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/component_predictions/mpnet_nt_predictions.pkl') and os.path.exists('data/component_predictions/mpnet_nt_not_ranked.pkl'):
    with open('data/component_predictions/mpnet_nt_predictions.pkl', 'rb') as f:
        predictions_mpnet_nt = pickle.load(f)
    with open('data/component_predictions/mpnet_nt_not_ranked.pkl', 'rb') as f:
        not_ranked_mpnet_nt = pickle.load(f)
else:
    predictions_mpnet_nt, not_ranked_mpnet_nt = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=0, distance_threshold=1, model="mpnet", top_n=n_candidates)
    predictions_mpnet_nt = rerank_queries_with_ollama(predictions_mpnet_nt, queries_dict, data_dict, max_candidates=n_candidates)
    with open('data/component_predictions/mpnet_nt_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_mpnet_nt, f)
    with open('data/component_predictions/mpnet_nt_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_mpnet_nt, f)
final_predictions = final_predictions | predictions_mpnet_nt
unresolved_queries &= set(not_ranked_mpnet_nt)
resolved_now = set(current_queries) - set(not_ranked_mpnet_nt)
print(f"MRR for resolved: {calculate_mrr(predictions_mpnet_nt, gold_dict)}")
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")
print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
print("RECALL: "+str(calculate_recall(predictions_mpnet_nt, gold_dict, data_dict, queries_dict, "no_recall/no_recall_mpnet_nt.txt")))


with open('predictions.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for key, value in final_predictions.items():
        writer.writerow([key, value])

df = pd.DataFrame(list(final_predictions.items()))
df.to_csv('training_predictions.csv', index=False)

print(f"MRR for final_predictions: {calculate_mrr(final_predictions, gold_dict)}")
mrr = calculate_mrr(final_predictions, gold_dict)
print("\nFinal Results:")
print(f"Training MRR - @1: {mrr[1]:.4f} @5: {mrr[5]:.4f} @10: {mrr[10]:.4f}")
print("RECALL: "+str(calculate_recall(final_predictions, gold_dict, data_dict, queries_dict, "no_recall/final.txt")))

save_query_results(unresolved_queries, queries_dict, gold_dict, data_dict, output_file="query_results.txt")