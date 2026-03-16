import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from io import BytesIO
import pickle
import gzip
import re
import pandas as pd
from collections import defaultdict

PATH_COLLECTION_DATA = '../data/subtask4b_collection_data.pkl'


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_paper_texts(id_ref_dict, output_file="../data/texts.pkl.gz"):
    id_text_dict = {}
    failure_stats = defaultdict(int)
    total_items = len(id_ref_dict)

    headers = {"User-Agent": "Mozilla/5.0"}

    for i, (id_, refs) in enumerate(id_ref_dict.items(), 1):
        doi = refs.get('doi')
        pmcid = refs.get('pmcid')
        pubmed_id = refs.get('pubmed_id')
        text = None
        success = False

        # Try DOI first
        if doi and pd.notna(doi):
            try:
                doi_url = f"https://doi.org/{doi}"
                response = requests.get(doi_url, headers=headers, allow_redirects=True)
                response.raise_for_status()
                final_url = response.url

                if final_url.lower().endswith(".pdf"):
                    try:
                        pdf_response = requests.get(final_url, headers=headers)
                        pdf_response.raise_for_status()
                        with BytesIO(pdf_response.content) as pdf_file:
                            reader = PdfReader(pdf_file)
                            text = " ".join(page.extract_text() or "" for page in reader.pages)
                        success = True
                    except:
                        failure_stats['doi_pdf_failed'] += 1
                else:
                    try:
                        html_response = requests.get(final_url, headers=headers)
                        html_response.raise_for_status()
                        soup = BeautifulSoup(html_response.text, 'html.parser')
                        for element in soup(["script", "style", "nav", "footer", "header"]):
                            element.decompose()
                        text = soup.get_text(separator=' ', strip=True)
                        success = True
                    except:
                        failure_stats['doi_html_failed'] += 1

                if text:
                    id_text_dict[id_] = clean_text(text)
                    print(f"Item {i}/{total_items} - Success (DOI): {id_}")
                    continue
                else:
                    failure_stats['doi_failed_after_redirect'] += 1
            except:
                failure_stats['doi_initial_failed'] += 1

        # Try PMCID next
        if pmcid and pd.notna(pmcid) and text is None:
            try:
                pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                html_response = requests.get(pmc_url, headers=headers)
                html_response.raise_for_status()
                soup = BeautifulSoup(html_response.text, 'html.parser')
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                text = soup.get_text(separator=' ', strip=True)

                if text:
                    id_text_dict[id_] = clean_text(text)
                    print(f"Item {i}/{total_items} - Success (PMCID): {id_}")
                    continue
                else:
                    # Try to get PDF from PMC
                    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
                    try:
                        pdf_response = requests.get(pdf_url, headers=headers)
                        pdf_response.raise_for_status()
                        with BytesIO(pdf_response.content) as pdf_file:
                            reader = PdfReader(pdf_file)
                            text = " ".join(page.extract_text() or "" for page in reader.pages)
                        if text:
                            id_text_dict[id_] = clean_text(text)
                            print(f"Item {i}/{total_items} - Success (PMC PDF): {id_}")
                            continue
                    except:
                        failure_stats['pmcid_pdf_failed'] += 1
                    failure_stats['pmcid_html_failed'] += 1
            except:
                failure_stats['pmcid_failed'] += 1

        # Try PubMed ID last
        if pubmed_id and pd.notna(pubmed_id) and text is None:
            try:
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
                html_response = requests.get(pubmed_url, headers=headers)
                html_response.raise_for_status()
                soup = BeautifulSoup(html_response.text, 'html.parser')
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                text = soup.get_text(separator=' ', strip=True)

                if text:
                    id_text_dict[id_] = clean_text(text)
                    print(f"Item {i}/{total_items} - Success (PubMed): {id_}")
                    continue
                else:
                    failure_stats['pubmed_no_text'] += 1
            except:
                failure_stats['pubmed_failed'] += 1

        # If all attempts failed
        if text is None:
            id_text_dict[id_] = pd.NA
            failure_stats['total_failures'] += 1
            print(f"Item {i}/{total_items} - Failed: {id_}")

    # Print failure statistics
    print("\nExtraction Statistics:")
    print(f"Total papers processed: {len(id_ref_dict)}")
    print(f"Successfully extracted: {len(id_text_dict) - failure_stats['total_failures']}")
    print(f"Failed extractions: {failure_stats['total_failures']}")
    print("\nFailure breakdown:")
    for reason, count in sorted(failure_stats.items()):
        if reason != 'total_failures':
            print(f"{reason}: {count}")

    # Save the dictionary with gzip and pickle
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(id_text_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return id_text_dict


# Load data
df = pd.read_pickle(PATH_COLLECTION_DATA)
id_ref_dict = df.set_index('cord_uid')[['doi', 'pmcid', 'pubmed_id']].to_dict(orient='index')
# Extract texts
text_dict = extract_paper_texts(id_ref_dict)
