
def calculate_mrr(predictions, gold_ids):
    reciprocal_ranks = {1: [], 5: [], 10: []}
    for query_id, gold_id in gold_ids.items():
        predicted_ids = predictions.get(query_id, [])
        if gold_id not in predicted_ids:
            continue
        rank = predicted_ids.index(gold_id) + 1
        for cutoff in reciprocal_ranks:
            if rank <= cutoff:
                reciprocal_ranks[cutoff].append(1.0 / rank)
    return {cutoff: sum(values) / len(values) if values else 0 for cutoff, values in reciprocal_ranks.items()}


def calculate_recall(results, goldfile, data_info, query_info, output_file=None):
    correct = 0
    total = 0
    missing_cases = []

    for query_id, correct_id in goldfile.items():
        if query_id not in results:
            continue
        total += 1
        if correct_id in results[query_id]:
            correct += 1
            continue
        top_ranked_id = results[query_id][0]
        top_paper = data_info.get(top_ranked_id, {"title": "N/A", "summary": "N/A"})
        missing_cases.append(
            {
                "query_id": query_id,
                "query_text": query_info.get(query_id, "N/A"),
                "correct_paper": data_info.get(correct_id, {"title": "N/A", "summary": "N/A"}),
                "top_ranked": {"title": top_paper["title"], "summary": top_paper["summary"]},
            }
        )

    if output_file and missing_cases:
        with open(output_file, 'w', encoding='utf-8') as f:
            for case in missing_cases:
                f.write(f"Query ID: {case['query_id']}\n")
                f.write(f"Query: {case['query_text']}\n\n")
                f.write("Correct Paper:\n")
                f.write(f"Title: {case['correct_paper']['title']}\n")
                f.write(f"Summary: {case['correct_paper']['summary']}\n\n")
                f.write("Top Ranked Paper:\n")
                f.write(f"Title: {case['top_ranked']['title']}\n")
                f.write(f"Summary: {case['top_ranked']['summary']}\n")
                f.write("\n" + "=" * 50 + "\n\n")

    return correct / total if total > 0 else 0.0


def prepare_text_inputs(df, use_summary=False):
    texts = []
    for _, row in df.iterrows():
        title = str(row["title"])
        summary = str(row.get("summary", ""))
        abstract = str(row.get("abstract", ""))
        if use_summary and summary and len(abstract) > 100:
            text = f"{title}: {summary}"
        elif not use_summary and len(abstract) > 100:
            text = f"{title}: {abstract}"
        else:
            text = title
        texts.append(text.strip())
    return texts


def clean_data(data_dict):
    cleaned = {}
    for key, value in data_dict.items():
        if value is None or (isinstance(value, float) and value != value):
            cleaned[key] = ""
        else:
            cleaned[key] = str(value)
    return cleaned


def save_query_results(query_ids, queries_dict, gold_dict, data_dict, output_file="query_results.txt"):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for query_id in query_ids:
                query_text = queries_dict.get(query_id, "QUERY TEXT NOT FOUND")
                correct_cord_uid = gold_dict.get(query_id) if gold_dict else None

                if correct_cord_uid is None:
                    correct_info = "CORRECT PAPER NOT FOUND\n"
                else:
                    paper_data = data_dict.get(correct_cord_uid, {})
                    title = paper_data.get("title", "TITLE NOT FOUND")
                    abstract = paper_data.get("abstract", "ABSTRACT NOT FOUND")
                    correct_info = f"CORRECT PAPER\n{correct_cord_uid}\n{title}\n{abstract}\n"

                f.write(f"QUERY\n{query_id}\n{query_text}\n")
                f.write(correct_info)
                f.write("-" * 50 + "\n\n")
                f.flush()

        print(f"Successfully saved results to {output_file}")
    except Exception as error:
        print(f"Error saving query results: {error}")


def fill_dict2(dict1, dict2, top_k):
    new_dict = {}
    for key, value in dict2.items():
        if len(value) < top_k:
            combined = value.copy()
            combined.extend(dict1.get(key, []))
            new_dict[key] = combined[:top_k]
            continue
        new_dict[key] = value
    return new_dict


def truncate_dict(dictionary, top_k=5):
    return {key: value[:top_k] for key, value in dictionary.items()}


def calculate_mrr_at5(predictions, ground_truth):
    reciprocal_ranks = []

    for query_id in ground_truth:
        if query_id not in predictions:
            reciprocal_ranks.append(0)
            continue

        predicted_list = predictions[query_id][:5]
        correct_ids = ground_truth[query_id]
        if not isinstance(correct_ids, (list, tuple, set)):
            correct_ids = [correct_ids]

        rank = None
        for i, pred_id in enumerate(predicted_list, start=1):
            if pred_id in correct_ids:
                rank = i
                break

        reciprocal_ranks.append(1.0 / rank if rank is not None else 0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

