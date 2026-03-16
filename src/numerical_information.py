import pickle
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import regex as re
import spacy
from word2number import w2n

from src.sentence_similarity import rank_by_semantic_similarity

PATH_COLLECTION_DATA = 'data/subtask4b_collection_data.pkl'
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def detect_numerical_information(texts, ids):
    result = {}
    for id, text in dict(zip(ids, texts)).items():
        numbers = []

        # Regex to match numbers (including commas, but not part of words like COVID-19)
        numeric_pattern = r'(?<![a-zA-Z0-9-])-?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![a-zA-Z])|(?<=\s)-?\d+(?:\.\d+)?(?=\s)'
        numeric_values = re.findall(numeric_pattern, text)
        # Remove commas and convert to float/int
        for num in numeric_values:
            cleaned_num = num.replace(',', '')  # Remove commas (e.g., "68,377" → "68377")
            if '.' in cleaned_num:
                numbers.append(float(cleaned_num))
            else:
                numbers.append(int(cleaned_num))

        # Regex for number words (unchanged)
        word_pattern = r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion)(?:\s+(?:and\s+)?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion)|-)+\b'

        for match in re.finditer(word_pattern, text, re.IGNORECASE):
            try:
                number = w2n.word_to_num(match.group())
                numbers.append(number)
            except ValueError:
                continue

        # Remove duplicates (while preserving order)
        seen = set()
        unique_numbers = []
        for num in numbers:
            if num not in seen:
                seen.add(num)
                unique_numbers.append(num)

        result[id] = unique_numbers

    return result


def categorize_and_sort_matches(dict1, dict2):
    dict2_sets = {id2: set(numbers) for id2, numbers in dict2.items()}

    has_matches = {}
    no_matches = []

    for id1, numbers1 in dict1.items():
        set1 = set(numbers1)
        matches = defaultdict(int)
        for id2, set2 in dict2_sets.items():
            shared = set1 & set2
            if shared:
                matches[id2] = len(shared)
        if matches:
            sorted_matches = sorted(matches.items(),
                                    key=lambda x: (-x[1], x[0]))
            has_matches[id1] = [id2 for id2, count in sorted_matches]
        else:
            no_matches.append(id1)
    return has_matches, no_matches

