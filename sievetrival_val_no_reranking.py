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

queries_summaries_df = pd.read_csv('data/dev_tweet_summaries.tsv', sep='\t', names=['id', 'summary'])
query_summaries_dict = dict(zip(queries_summaries_df['id'], queries_summaries_df['summary']))

all_texts_with_summary = prepare_text_inputs(df_collection, use_summary=True)
all_texts_with_abstract = prepare_text_inputs(df_collection, use_summary=False)
all_data_ids = df_collection['cord_uid'].tolist()
id_to_abstract = {uid: text for uid, text in zip(all_data_ids, all_texts_with_abstract)}
id_to_summary = {uid: text for uid, text in zip(all_data_ids, all_texts_with_summary)}

whole_papers_dict.update({k: v for k, v in id_to_abstract.items() if k not in whole_papers_dict})

# Load test data (assuming it follows the same format as train/dev)
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

print(f"Starting pipeline with {len(queries)} queries")

final_predictions = {}

print("\n Component: Title Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/title_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/title_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/title_predictions.pkl', 'rb') as f:
        predictions_td = pickle.load(f)
    with open('data/nrr_dev_component_predictions/title_not_ranked.pkl', 'rb') as f:
        not_ranked_td = pickle.load(f)
else:
    predictions_td, not_ranked_td = rank_by_title(current_queries_dict, data_dict)
    with open('data/nrr_dev_component_predictions/title_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_td, f)
    with open('data/nrr_dev_component_predictions/title_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_td, f)
final_predictions= final_predictions | predictions_td
unresolved_queries &= set(not_ranked_td)
resolved_now = set(current_queries) - set(not_ranked_td)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

print(len(final_predictions))

print("\n Component: Quote Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/quote_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/quote_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/quote_predictions.pkl', 'rb') as f:
        predictions_qd = pickle.load(f)
    with open('data/nrr_dev_component_predictions/quote_not_ranked.pkl', 'rb') as f:
        not_ranked_qd = pickle.load(f)
else:
    predictions_qd, not_ranked_qd = rank_by_quotes(current_queries_dict, all_data_ids, data_dict, texts_file="data/texts.pkl.gz")
    predictions_qd = rerank_queries_with_ollama(predictions_qd, queries_dict, data_dict, max_candidates=n_candidates)
    with open('data/nrr_dev_component_predictions/quote_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_qd, f)
    with open('data/nrr_dev_component_predictions/quote_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_qd, f)
final_predictions= final_predictions | predictions_qd
unresolved_queries &= set(not_ranked_qd)
resolved_now = set(current_queries) - set(not_ranked_qd)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

print(len(final_predictions))

print("\n Component: Journal Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/journal_predictions_no_reranking.pkl') and os.path.exists('data/nrr_dev_component_predictions/journal_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/journal_predictions_no_reranking.pkl', 'rb') as f:
        predictions_jd = pickle.load(f)
    with open('data/nrr_dev_component_predictions/journal_not_ranked.pkl', 'rb') as f:
        not_ranked_jd = pickle.load(f)
else:
    predictions_jd, not_ranked_jd = rank_by_journal(current_queries_dict, data_dict)
    #predictions_jd = rerank_queries_with_ollama(predictions_jd, queries_dict, data_dict, max_candidates=n_candidates)
    with open('data/nrr_dev_component_predictions/journal_predictions_no_reranking.pkl', 'wb') as f:
        pickle.dump(predictions_jd, f)
    with open('data/nrr_dev_component_predictions/journal_not_ranked_no_reranking.pkl', 'wb') as f:
        pickle.dump(not_ranked_jd, f)
final_predictions = final_predictions | predictions_jd
unresolved_queries &= set(not_ranked_jd)
resolved_now = set(current_queries) - set(not_ranked_jd)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

print(len(final_predictions))

print("\n Component: Author Detection")
current_queries = list(unresolved_queries)
current_queries_dict = {k: queries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/author_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/author_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/author_predictions.pkl', 'rb') as f:
        predictions_ad = pickle.load(f)
    with open('data/nrr_dev_component_predictions/author_not_ranked.pkl', 'rb') as f:
        not_ranked_ad = pickle.load(f)
else:
    predictions_ad, not_ranked_ad = match_authors_to_queries(data_dict, current_queries_dict)
    #predictions_ad = rerank_queries_with_ollama(predictions_ad, queries_dict, data_dict, max_candidates=n_candidates)
    with open('data/nrr_dev_component_predictions/author_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_ad, f)
    with open('data/nrr_dev_component_predictions/author_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_ad, f)
final_predictions.update(predictions_ad)
unresolved_queries &= set(not_ranked_ad)
resolved_now = set(current_queries) - set(not_ranked_ad)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

print(len(final_predictions))

print("\n Component: Summary Similarity")
current_queries = list(unresolved_queries)
current_queries_dict = {k: query_summaries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/summary_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/summary_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/summary_predictions.pkl', 'rb') as f:
        predictions_sus = pickle.load(f)
    with open('data/nrr_dev_component_predictions/summary_not_ranked.pkl', 'rb') as f:
        not_ranked_sus = pickle.load(f)
else:
    predictions_sus, not_ranked_sus = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold, distance_threshold=distance_threshold, top_n=n_candidates)
    with open('data/nrr_dev_component_predictions/summary_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_sus, f)
    with open('data/nrr_dev_component_predictions/summary_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_sus, f)
final_predictions = final_predictions | predictions_sus
unresolved_queries &= set(not_ranked_sus)
resolved_now = set(current_queries) - set(not_ranked_sus)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

print(len(final_predictions))

print("\n Component: Summary Similarity GTE")
current_queries = list(unresolved_queries)
current_queries_dict = {k: query_summaries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/gte_summarypredictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/gte_summarynot_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/gte_summarypredictions.pkl', 'rb') as f:
        predictions_gte = pickle.load(f)
    with open('data/nrr_dev_component_predictions/gte_summarynot_ranked.pkl', 'rb') as f:
        not_ranked_gte = pickle.load(f)
else:
    predictions_gte, not_ranked_gte = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold_gte, distance_threshold=distance_threshold_gte, top_n=n_candidates, model="gte")
    with open('data/nrr_dev_component_predictions/gte_summarypredictions.pkl', 'wb') as f:
        pickle.dump(predictions_gte, f)
    with open('data/nrr_dev_component_predictions/gte_summarynot_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_gte, f)
final_predictions = final_predictions | predictions_gte
unresolved_queries &= set(not_ranked_gte)
resolved_now = set(current_queries) - set(not_ranked_gte)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

print(calculate_mrr_at5(final_predictions, gold_dict))
print(len(final_predictions))

# print("\n Component: Summary Similarity BGE")
# current_queries = list(unresolved_queries)
# current_queries_dict = {k: query_summaries_dict[k] for k in current_queries}
# if os.path.exists('data/nrr_dev_component_predictions/bge_summary_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/bge_summary_not_ranked.pkl'):
#     with open('data/nrr_dev_component_predictions/bge_summary_predictions.pkl', 'rb') as f:
#         predictions_bge = pickle.load(f)
#     with open('data/nrr_dev_component_predictions/bge_summary_not_ranked.pkl', 'rb') as f:
#         not_ranked_bge = pickle.load(f)
# else:
#     predictions_bge, not_ranked_bge = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold, distance_threshold=distance_threshold, top_n=n_candidates, model="bge")
#     with open('data/nrr_dev_component_predictions/bge_summary_predictions.pkl', 'wb') as f:
#         pickle.dump(predictions_bge, f)
#     with open('data/nrr_dev_component_predictions/bge_summary_not_ranked.pkl', 'wb') as f:
#         pickle.dump(not_ranked_bge, f)
# final_predictions = final_predictions | predictions_bge
# unresolved_queries &= set(not_ranked_bge)
# resolved_now = set(current_queries) - set(not_ranked_bge)
# print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

print(calculate_mrr_at5(final_predictions, gold_dict))
print(len(final_predictions))

print("\n Component: BM25 on Abstracts")
current_queries = list(unresolved_queries)
current_queries_dict = {k: query_summaries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/bm25_abstracts_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/bm25_abstracts_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/bm25_abstracts_predictions.pkl', 'rb') as f:
        predictions_bm25 = pickle.load(f)
    with open('data/nrr_dev_component_predictions/bm25_abstracts_not_ranked.pkl', 'rb') as f:
        not_ranked_bm25 = pickle.load(f)
else:
    predictions_bm25, not_ranked_bm25 = rank_all_queries(current_queries_dict, id_to_abstract, threshold=bm25_abstracts_threshold, top_k=n_candidates)
    with open('data/nrr_dev_component_predictions/bm25_abstracts_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_bm25, f)
    with open('data/nrr_dev_component_predictions/bm25_abstracts_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_bm25, f)
overlap = set(predictions_sus.keys()) & set(predictions_bm25.keys())
final_predictions = final_predictions | predictions_bm25
unresolved_queries &= set(not_ranked_bm25)
resolved_now = set(current_queries) - set(not_ranked_bm25)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

print(len(final_predictions))

print("\n Component: Scientific Similarity SPECTER")
current_queries = list(unresolved_queries)
current_queries_dict = {k: query_summaries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/specter_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/specter_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/specter_predictions.pkl', 'rb') as f:
        predictions_specs = pickle.load(f)
    with open('data/nrr_dev_component_predictions/specter_not_ranked.pkl', 'rb') as f:
        not_ranked_specs = pickle.load(f)
else:
    predictions_specs, not_ranked_specs = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold_spc, distance_threshold=distance_threshold_spc, model="specter", top_n=n_candidates)
    with open('data/nrr_dev_component_predictions/specter_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_specs, f)
    with open('data/nrr_dev_component_predictions/specter_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_specs, f)
final_predictions= final_predictions | predictions_specs
unresolved_queries &= set(not_ranked_specs)
resolved_now = set(current_queries) - set(not_ranked_specs)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")


print(len(final_predictions))

print("\n Component: Summary Similarity MPNET")
current_queries = list(unresolved_queries)
current_queries_dict = {k: query_summaries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/mpnet_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/mpnet_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/mpnet_predictions.pkl', 'rb') as f:
        predictions_mpnet = pickle.load(f)
    with open('data/nrr_dev_component_predictions/mpnet_not_ranked.pkl', 'rb') as f:
        not_ranked_mpnet = pickle.load(f)
else:
    predictions_mpnet, not_ranked_mpnet = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=similarity_threshold_mpnet, distance_threshold=distance_threshold_mpnet, model="mpnet", top_n=n_candidates)
    with open('data/nrr_dev_component_predictions/mpnet_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_mpnet, f)
    with open('data/nrr_dev_component_predictions/mpnet_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_mpnet, f)
final_predictions = final_predictions | predictions_mpnet
unresolved_queries &= set(not_ranked_mpnet)
resolved_now = set(current_queries) - set(not_ranked_mpnet)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")


print(len(final_predictions))


print("\n Component: Unique Token Linking")
current_queries = list(unresolved_queries)
current_queries_dict = {k: query_summaries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/token_linking_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/token_linking_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/token_linking_predictions.pkl', 'rb') as f:
        predictions_untr = pickle.load(f)
    with open('data/nrr_dev_component_predictions/token_linking_not_ranked.pkl', 'rb') as f:
        not_ranked_untr = pickle.load(f)
else:
    predictions_untr, not_ranked_untr = find_queries_with_seldom_tokens(current_queries_dict, id_to_abstract)
    with open('data/nrr_dev_component_predictions/token_linking_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_untr, f)
    with open('data/nrr_dev_component_predictions/token_linking_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_untr, f)
final_predictions = final_predictions | predictions_untr
unresolved_queries &= set(not_ranked_untr)
resolved_now = set(current_queries) - set(not_ranked_untr)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

print(len(final_predictions))

print("\n Component: Fill with Summary Similarity MPNET No Thresholds")
if os.path.exists('data/nrr_dev_component_predictions/mpnet_ntf_predictions.pkl'):
    with open('data/nrr_dev_component_predictions/mpnet_ntf_predictions.pkl', 'rb') as f:
        all_predictions_mpnet_ntf = pickle.load(f)
else:
    all_predictions_mpnet_ntf, _ = rank_by_semantic_similarity(query_summaries_dict, all_texts_with_summary,
                                                                        all_data_ids, similarity_threshold=0,
                                                                        distance_threshold=0, model="mpnet",
                                                                        top_n=n_candidates)
    with open('data/nrr_dev_component_predictions/mpnet_ntf_predictions.pkl', 'wb') as f:
        pickle.dump(all_predictions_mpnet_ntf, f)

final_predictions = fill_dict2(all_predictions_mpnet_ntf, final_predictions, n_candidates)


print(len(final_predictions))

print("\n Component: Summary Summary Similarity MPNET Low Thresholds")
current_queries = list(unresolved_queries)
current_queries_dict = {k: query_summaries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/mpnet_snt_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/mpnet_snt_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/mpnet_snt_predictions.pkl', 'rb') as f:
        predictions_mpnet_snt = pickle.load(f)
    with open('data/nrr_dev_component_predictions/mpnet_snt_not_ranked.pkl', 'rb') as f:
        not_ranked_mpnet_snt = pickle.load(f)
else:
    predictions_mpnet_snt, not_ranked_mpnet_snt = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=0.7, distance_threshold=0, model="mpnet", top_n=n_candidates)
    #predictions_mpnet_nt = rerank_queries_with_ollama(predictions_mpnet_nt, queries_dict, data_dict, max_candidates=n_candidates)
    with open('data/nrr_dev_component_predictions/mpnet_snt_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_mpnet_snt, f)
    with open('data/nrr_dev_component_predictions/mpnet_snt_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_mpnet_snt, f)
final_predictions = final_predictions | predictions_mpnet_snt
unresolved_queries &= set(not_ranked_mpnet_snt)
resolved_now = set(current_queries) - set(not_ranked_mpnet_snt)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

for id in unresolved_queries:
    print(queries_dict[id])
    print(id_to_summary[gold_dict[id]])

print("\n Component: Final Summary Summary Similarity bge No Thresholds")
current_queries = list(unresolved_queries)
print(len(current_queries))
current_queries_dict = {k: query_summaries_dict[k] for k in current_queries}
if os.path.exists('data/nrr_dev_component_predictions/bge_final_predictions.pkl') and os.path.exists('data/nrr_dev_component_predictions/bge_final_not_ranked.pkl'):
    with open('data/nrr_dev_component_predictions/bge_final_predictions.pkl', 'rb') as f:
        predictions_bge_final = pickle.load(f)
    with open('data/nrr_dev_component_predictions/bge_final_not_ranked.pkl', 'rb') as f:
        not_ranked_bge_final = pickle.load(f)
else:
    predictions_bge_final, not_ranked_bge_final = rank_by_semantic_similarity(current_queries_dict, all_texts_with_summary, all_data_ids, similarity_threshold=0, distance_threshold=0, model="mpnet", top_n=n_candidates)
    #predictions_bge_final = rerank_queries_with_ollama(predictions_bge_final, queries_dict, data_dict, max_candidates=n_candidates, use_lm=True)
    with open('data/nrr_dev_component_predictions/bge_final_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_bge_final, f)
    with open('data/nrr_dev_component_predictions/bge_final_not_ranked.pkl', 'wb') as f:
        pickle.dump(not_ranked_bge_final, f)
final_predictions = final_predictions | predictions_bge_final
unresolved_queries &= set(not_ranked_bge_final)
resolved_now = set(current_queries) - set(not_ranked_bge_final)
print(f"Resolved queries: {total_queries - len(unresolved_queries)}/{total_queries}")

final_predictions = truncate_dict(final_predictions)
print(calculate_mrr_at5(final_predictions, gold_dict))


# for key, value in final_predictions.items():
#     print(queries_dict[key])
#     print(id_to_summary[value[0]])
#

with open('FINAL_VAL_PREDICTIONS.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(["post_id", "preds"])
    for key, value in final_predictions.items():
        writer.writerow([key, value])

print("\nFinal Results:")
print(f"Total queries processed: {total_queries}")
print(f"Unresolved queries: {len(unresolved_queries)}")

#save_query_results(unresolved_queries, queries_dict, None, data_dict, output_file="query_results.txt")