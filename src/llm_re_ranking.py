from ollama import generate

from src.sentence_similarity import rank_to_one_query_semantic_similarity

RERANKING_MODEL = "llama3.1:8b"
MAX_COMPARISON_ITEMS = 5
TOP_K_PER_ROUND = 2


def rerank_queries_with_ollama(query_data, query_texts, data_info, max_candidates, use_lm=False):
    reranked_results = {}
    if not use_lm:
        for query_id, data_ids in query_data.items():
            data_ids = rank_to_one_query_semantic_similarity(query_texts[query_id], data_ids, data_info, top_n=max_candidates)
            reranked_results[query_id] = data_ids
    else:
        for query_id, data_ids in query_data.items():
            print(query_id)
            items_to_rank = [
                (data_id, f"ID: {data_id}\n" + "\n".join(
                    f"{k}: {v}" for k, v in data_info[data_id].items() if k in ["title", "journal", "authors", "summary"]
                ))
                for data_id in data_ids if data_id in data_info
            ]
            if not items_to_rank:
                reranked_results[query_id] = []
                continue
            if len(items_to_rank) > MAX_COMPARISON_ITEMS:
                final_ranking = tournament_style_ranking(items_to_rank, query_texts[query_id])
            else:
                final_ranking = rank_items(items_to_rank, query_texts[query_id])
            if final_ranking:
                reranked_results[query_id] = final_ranking
            else:
                reranked_results[query_id] = data_ids
    return reranked_results


def tournament_style_ranking(items, query_text):
    winners = items.copy()
    while len(winners) > MAX_COMPARISON_ITEMS:
        chunks = [winners[i:i + MAX_COMPARISON_ITEMS] for i in range(0, len(winners), MAX_COMPARISON_ITEMS)]
        winners = []
        for chunk in chunks:
            ranked_chunk_ids = rank_items(chunk, query_text)
            top_k_ids = ranked_chunk_ids[:TOP_K_PER_ROUND]
            winners.extend([item for item in chunk if item[0] in top_k_ids])

    final_ranked_ids = rank_items(winners, query_text)
    ranked_set = set(final_ranked_ids)
    remaining = [item[0] for item in items if item[0] not in ranked_set]

    return final_ranked_ids + remaining


def rank_items(items, query_text):
    if len(items) <= 1:
        return [items[0][0]] if items else []

    id_to_item = {item[0]: item for item in items}

    prompt = f"Rank these by similarity to query: {query_text}\nItems:\n"
    prompt += "\n".join(f"ITEM: {item[0]}\n{item[1]}" for item in items)
    prompt += "\n\nExplain your ranking without listing the items with numbers."
    prompt += f" AT THE END OF YOUR RESPONSE, return the ranked item IDs in order, comma-separated. Example: 9ljou3ag, vpih1wvs, 55yxh5er"
    response = generate(model=RERANKING_MODEL, prompt=prompt)['response']
    try:
        last_line = None
        for line in reversed(response.split('\n')):
            if any(item_id.strip() in id_to_item for item_id in line.split(',')):
                last_line = line.strip()
                break

        if last_line:
            ranked_ids = [id_.strip() for id_ in last_line.split(',') if id_.strip() in id_to_item]
            if len(ranked_ids) == len(items):
                return ranked_ids
            missing_ids = [item[0] for item in items if item[0] not in ranked_ids]
            return ranked_ids + missing_ids

        print("Warning: Could not parse ranked IDs, falling back to original order.")
        return [item[0] for item in items]

    except Exception as e:
        print(f"Error parsing ranking: {e}")
        return [item[0] for item in items]
