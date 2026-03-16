import regex as re
from collections import defaultdict
import requests
import spacy


def create_name_variants(author, stopwords, words, custom_stop_words):
    author = author.strip()
    if not author:
        return []

    parts = [p.strip() for p in re.split(r'[,/]', author) if p.strip()]
    variants = []

    for part in parts:
        name_parts = part.split()
        if len(name_parts) >= 2:
            firstname = ' '.join(name_parts[:-1])
            lastname = name_parts[-1]
            if len(firstname) > 1 and len(
                    lastname) > 1 and lastname.lower() not in stopwords and lastname.lower() not in words and lastname.lower() not in custom_stop_words:
                variants.append(f"{firstname} {lastname}".lower())
                variants.append(f"{lastname} {firstname}".lower())
            if len(lastname) > 3 and lastname.lower() not in stopwords and lastname.lower() not in words and lastname.lower() not in custom_stop_words:
                variants.append(lastname.lower())
        elif part and len(
                part) > 3 and part.lower() not in stopwords and part.lower() not in words and part.lower() not in custom_stop_words:
            variants.append(part.lower())
    return list(set(variants))


def sanitize(s):
    return re.sub('[^\w\s]', ' ', s)


def match_authors_to_queries(cord_data, query_data):
    nlp = spacy.load("en_core_web_sm")
    stop_words = nlp.Defaults.stop_words
    words = requests.get(
        "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt").text.splitlines()
    author_to_cords = defaultdict(set)
    custom_stop_words = ["covid-19", "than the", "fauci", "e.g.", "socio", "lancet", "harms", "vivo",
                         "jama", "coma", "parkinson", "achilles", "candida", "trump", "breeze", "prudence",
                         "group study", "youngster", "biobank", "kawasaki", "forecasting", "padova", "harsh", "priori",
                         "shields", "gram", "catching", "yawn", "stark", "gamble", "grimes", "sheilds", "sera", "fuss",
                         "stalin", "milano"
                         ]

    for cord_id, data in cord_data.items():
        authors = data.get('authors', '')
        if not authors or isinstance(authors, float):
            continue

        raw_authors = re.split(r'[;,]', authors)
        for author in raw_authors:
            for variant in create_name_variants(author, stop_words, words, custom_stop_words):
                author_to_cords[variant].add(cord_id)

    predictions = {}
    not_ranked = []

    for query_id, query_text in query_data.items():
        query_lower = f" {sanitize(query_text.lower())} "
        found_authors = set()

        for author in author_to_cords:
            author_with_spaces = f" {sanitize(author)} "
            if author_with_spaces in query_lower:
                found_authors.add(author)

        if not found_authors:
            not_ranked.append(query_id)
            continue

        matching_cords = None
        for author in found_authors:
            if matching_cords is None:
                matching_cords = set(author_to_cords[author])
            else:
                matching_cords.intersection_update(author_to_cords[author])

        if matching_cords:
            predictions[query_id] = list(matching_cords)
        else:
            not_ranked.append(query_id)

    return predictions, not_ranked

# queries_dict = {
#     "2068": "Neutralisationstests von Corman/Drosten: \"Neutralisation testing is crucial to exclude cross-reactive antibodies against common human coronaviruses. We have developed a highly sensitive plaque-reduction neutralisation assay (PRNT)",
# "3021": "tässä lasten tartunnoista \"in an objective way\": \"kids are at a similar risk of infection to the general public, although less likely to have severe symptoms; hence they should be considered in analyses of transmission and control.",
# "42": "xu, x. s., bulger, e. a.,..., akbari, o. s., marshall, j. m. and bier, e. (2020). active genetic neutralizing elements for halting or eradicating gene drives. mol cell.	9n42e9k3",
# "44": "bears repeating: ""no medication definitely reduced mortality, or reduced initiation of ventilation or hospitalization duration. these #remdesivir #hydroxychloroquine #lopinavir &amp; #interferon regimens had little or no effect on hospitalized patients w/#covid19 """
# }
#
# PATH_COLLECTION_DATA = '../data/subtask4b_collection_data.pkl'
# df = pd.read_pickle(PATH_COLLECTION_DATA)
# id_author_dict = df.set_index('cord_uid')[['authors']].to_dict(orient='index')
#
# results, not_matched = match_authors_to_queries(id_author_dict, queries_dict)
# print(results)
# print(not_matched)
