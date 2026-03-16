import math
import pandas as pd
from collections import defaultdict

import spacy


def generate_name_variations(author, stop_words):
    author = author.strip()
    variations = {author}
    if ',' in author:
        lastname, firstname = (part.strip() for part in author.split(',', 1))
        variations.add(f"{firstname} {lastname}")
        if len(lastname) > 1 and lastname[1] != '.' and lastname not in stop_words:
            variations.add(lastname)
    return variations


def match_authors_to_queries(data_dict, query_dict):
    nlp = spacy.load("en_core_web_sm")
    stop_words = nlp.Defaults.stop_words
    author_to_data = defaultdict(list)
    data_id_to_total_authors = {}

    for data_id, data_info in data_dict.items():
        authors_str = data_info.get("authors", "")
        if authors_str is None or (isinstance(authors_str, float) and math.isnan(authors_str)):
            authors = set()
        else:
            authors = set()
            for author in str(authors_str).split(';'):
                author = author.strip()
                if author:
                    authors.update(generate_name_variations(author, stop_words))

        data_id_to_total_authors[data_id] = len(authors)
        for author in authors:
            author_to_data[author].append((data_id, len(authors)))

    result = {}
    no_match_queries = []

    for query_id, query_text in query_dict.items():
        matching_data = defaultdict(int)
        found_match = False

        for author, data_list in author_to_data.items():
            if " "+author+" " in query_text:
                found_match = True
                for data_id, total_authors in data_list:
                    matching_data[data_id] += 1

        if not found_match:
            no_match_queries.append(query_id)
            continue

        sorted_data_ids = sorted(
            matching_data.keys(),
            key=lambda x: (-matching_data[x], -data_id_to_total_authors[x], x)
        )

        result[query_id] = sorted_data_ids

    return result, no_match_queries



# queries_dict = {
#     "2068": "Neutralisationstests von Corman/Drosten: \"Neutralisation testing is crucial to exclude cross-reactive antibodies against common human coronaviruses. We have developed a highly sensitive plaque-reduction neutralisation assay (PRNT)",
# "3021": "tässä lasten tartunnoista \"in an objective way\": \"kids are at a similar risk of infection to the general public, although less likely to have severe symptoms; hence they should be considered in analyses of transmission and control."}
#
# PATH_COLLECTION_DATA = '../data/subtask4b_collection_data.pkl'
# df = pd.read_pickle(PATH_COLLECTION_DATA)
# id_author_dict = df.set_index('cord_uid')[['authors']].to_dict(orient='index')
#
# results, not_matched = match_authors_to_queries(id_author_dict, queries_dict)
# print(results)
# print(not_matched)
