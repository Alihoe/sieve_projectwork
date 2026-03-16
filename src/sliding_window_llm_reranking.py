import requests
import json

MODEL = "llama3.1:8b"



def sliding_window_rerank(query, data_ids, id_to_text, window_size=WINDOW_SIZE, top_k=TOP_K_PER_WINDOW):
    ranked_data_ids = []

    for i in range(0, len(data_ids), window_size):
        window_ids = data_ids[i:i + window_size]
        window_texts = [id_to_text[data_id] for data_id in window_ids]

        prompt = f"""Rank these passages by relevance to the query: "{query}".
        Return ONLY the original indices (0-based) of the top {top_k} passages in order, like [2, 0, 1].
        Passages:
        """
        for idx, text in enumerate(window_texts):
            prompt += f"\n{idx}. {text[:200]}"

        response = ollama_generate(prompt, model=MODEL)["response"]

        try:
            ranked_indices = json.loads(response.strip())
            if isinstance(ranked_indices, list) and len(ranked_indices) > 0:
                ranked_data_ids.extend([window_ids[idx] for idx in ranked_indices[:top_k]])
        except:
            ranked_data_ids.extend(window_ids[:top_k])

    return ranked_data_ids


def batch_rerank(query_to_data_ids, id_to_text):
    results = {}
    for query_id, data_ids in query_to_data_ids.items():
        results[query_id] = sliding_window_rerank(query_id, data_ids, id_to_text)
    return results