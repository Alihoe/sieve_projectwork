def rank_by_title(query_dict, data_dict):
    predictions = {}
    queries_without_titles = []

    for query_id, query_text in query_dict.items():
        query_text_lower = query_text.lower()
        matches = [
            data_id
            for data_id, data in data_dict.items()
            if "title" in data and data["title"] and data["title"].lower() in query_text_lower
        ]
        if matches:
            predictions[query_id] = matches
        else:
            queries_without_titles.append(query_id)

    return predictions, queries_without_titles
