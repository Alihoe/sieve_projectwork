from collections import defaultdict
import re
import requests
from functools import lru_cache


@lru_cache(maxsize=1)
def get_common_words():
    words_url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt"
    response = requests.get(words_url)
    words = response.text.splitlines()
    journal_words = ["in vivo", "inflammation", "immunity", "public health", "global health", "microbiome", "nutrients",
                     "hepatology", "gut", "radiology", "transfusion", "virology", "multiple sclerosis", "pathogens",
                     "placenta", "headache", "microorganisms", "neurology", "epidemics", "hypertension", "hippocampus",
                     "virulence", "neuron", "pediatrics", "dna repair", "tobacco control", "systematic reviews",
                     "ophthalmology", "pathophysiology", "cancer cell", "transplantation", "epidemiology",
                     "neuropsychopharmacology", "scientific reports", "sports medicine", "bone marrow transplant",
                     "current biology", "dna repair", "drug and alcohol dependence", "nitric oxide", "brain injury",
                     "multiple sclerosis", "health policy", "amino acids", "leukemia", "resuscitation", "gut microbes",
                     "atherosclerosis", "critical care medicine", "bioinformatics", "clinical infectious diseases",
                     "metabolites", "seizure", "neurology", "endocrine", "diabetes care", "biomedicines",
                     "cell metabolism", "respiration", "antiviral therapy", "immunobiology", "indoor air", "policing",
                     "criminology", "psychological science"]
    words += [word.lower() for word in journal_words]
    return set(words)


def rank_by_journal(query_dict, data_dict):
    pipeline_phrases = ["published in", "released in"]
    words = get_common_words()

    journal_to_cords = defaultdict(set)
    all_journals = set()

    for cord_id, data in data_dict.items():
        journal = data.get('journal', '')
        if journal and not isinstance(journal, float):
            journal_lower = journal.lower()
            journal_to_cords[journal_lower].add(cord_id)
            all_journals.add(journal_lower)

    patterns = {}
    for journal in all_journals:
        if ' ' not in journal:
            pattern = re.compile(rf'(?<!\w){re.escape(journal)}(?!\w)')
        else:
            patterns_list = [
                rf'^{re.escape(journal)}(?=\W)',
                rf'(?<=\W){re.escape(journal)}(?=\W)',
                rf'(?<=\W){re.escape(journal)}$'
            ]
            pattern = re.compile('|'.join(patterns_list))
        patterns[journal] = pattern

    predictions = {}
    not_ranked = []

    pipeline_patterns = []
    for phrase in pipeline_phrases:
        pipeline_patterns.append((
            re.compile(re.escape(phrase)),
            re.compile(rf'{re.escape(phrase)}\s+(\w+)')
        ))

    for query_id, query_text in query_dict.items():
        query_lower = query_text.lower()
        found_cords_set = set()
        found_journals = []

        for journal, pattern in patterns.items():
            if journal not in words and pattern.search(query_lower):
                found_journals.append(journal)
                found_cords_set.update(journal_to_cords[journal])

        if not found_cords_set:
            for phrase_re, next_word_re in pipeline_patterns:
                if phrase_re.search(query_lower):
                    match = next_word_re.search(query_lower)
                    if match:
                        next_word = match.group(1).lower()
                        if next_word in all_journals:
                            found_journals.append(next_word)
                            found_cords_set.update(journal_to_cords[next_word])

        if found_cords_set:
            # print(f'Query: {query_text}')
            # print(f'Found Journals: {found_journals}')
            # print(data_dict[list(found_cords_set)[0]]["journal"])
            max_len = max(len(x) for x in found_cords_set)
            prediction_list = [s for s in found_cords_set if len(s) == max_len]
            predictions[query_id] = prediction_list
        else:
            not_ranked.append(query_id)

    # for key, value in predictions.items():
    #     print(query_dict[key])
    #     print(data_dict[value[0]]["journal"])
    return predictions, not_ranked