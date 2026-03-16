import numpy as np
import spacy


def extract_wikidata_ids(texts):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    results = []
    for text in texts:
        doc = nlp(text)
        ids = [ent.get_id() for ent in doc._.linkedEntities]
        results.append(ids)
    return results


def calculate_avg_matches(entities_a, entities_b, ids_b, gold_ids):
    b_entity_map = {b_id: set(entities) for b_id, entities in zip(ids_b, entities_b)}
    matches = [len(set(a_entities) & b_entity_map[gold_id]) for a_entities, gold_id in zip(entities_a, gold_ids)]
    avg = np.mean(matches)
    std_dev = np.std(matches) if len(matches) > 1 else 0.0
    return avg + std_dev


def find_top_matches_with_threshold(entities_a, ids_a, entities_b, ids_b, threshold):
    result = []
    weak_matches = []

    b_entity_sets = {b_id: set(entities) for b_id, entities in zip(ids_b, entities_b)}

    for a_entities, a_id in zip(entities_a, ids_a):
        a_set = set(a_entities)
        max_match = 0
        scores = []

        for b_id, b_set in b_entity_sets.items():
            intersection = len(a_set & b_set)
            scores.append((intersection, b_id))
            if intersection > max_match:
                max_match = intersection

        scores.sort(reverse=True)
        top_matches = [b_id for _, b_id in scores[:5]]
        result.append(top_matches)

        if max_match < threshold:
            weak_matches.append(a_id)

    return result, weak_matches


def update_predictions_with_top_matches(predictions, not_ranked, entities_a, ids_a, entities_b, ids_b, threshold,
                                        top_n=3000):
    a_id_to_index = {a_id: idx for idx, a_id in enumerate(ids_a)}
    b_id_to_entities = {b_id: set(entities) for b_id, entities in zip(ids_b, entities_b)}
    updated_predictions = [lst.copy() for lst in predictions]
    weak_matches = []

    for a_id in not_ranked:
        if a_id not in a_id_to_index:
            continue

        idx = a_id_to_index[a_id]
        a_entities = set(entities_a[idx])
        candidate_b_ids = predictions[idx]
        scores = []
        max_match = 0

        for b_id in candidate_b_ids:
            if b_id not in b_id_to_entities:
                continue

            b_entities = b_id_to_entities[b_id]
            intersection = len(a_entities & b_entities)
            scores.append((intersection, b_id))
            if intersection > max_match:
                max_match = intersection

        scores.sort(reverse=True, key=lambda x: x[0])
        updated_predictions[idx] = [b_id for score, b_id in scores[:top_n]]

        if max_match < threshold:
            weak_matches.append(a_id)

    return updated_predictions, weak_matches

