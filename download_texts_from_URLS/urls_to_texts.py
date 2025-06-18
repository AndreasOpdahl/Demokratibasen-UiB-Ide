"""
Uses code from Demokratibasen to convert URLS to text files.

This script intends to mimic the conversion done by Demokratibasen as of May 2025.
"""

from datetime import datetime
import json
import logging
import mimetypes
import os
import requests
import re
from requests.exceptions import Timeout, ReadTimeout, ChunkedEncodingError
import unicodedata

import magic
import pandas as pd

from pdfextraction import PdfExtraction, DocxExtraction


INPUT_FILE = "dokumenter.csv"
OUTPUT_FILE = "dokumenter.jsonl"


def get_documents(INPUT_FILE):
    doc_df = pd.read_csv(INPUT_FILE)
    return doc_df.to_dict(orient='records')


def get_logger(logger_name, log_level=logging.DEBUG):
    logger = logging.getLogger(logger_name)

    today = datetime.now().strftime("%d-%m-%Y")
    log_dir = f"logs/{today}"
    log_path = os.path.join(log_dir, f"{logger_name}.log")

    # Delete logs older than 5 days
    for dirname in os.listdir(f"{os.getcwd()}/logs"):
        dir_date_str = dirname.split('.')[0]
        try:
            dir_date = datetime.strptime(dir_date_str, "%d-%m-%Y")
            if (datetime.now() - dir_date).days > 5:
                dir_path = os.path.join(f"{os.getcwd()}/logs", dirname)
                for filename in os.listdir(dir_path):
                    os.remove(os.path.join(dir_path, filename))
                os.rmdir(dir_path)
        except ValueError:
            continue

    if not os.path.exists(log_path):
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def fetch_document(logger, url):
    try:
        response = requests.get(url=url, timeout=(15, 60))
        response_size_kb = round(len(response.content) / 1024)
        elapsed_time_seconds = round(response.elapsed.total_seconds(), 2)
        logger.info(
            f"GET [{response.status_code}]: {url} {elapsed_time_seconds} {response_size_kb}"
        )
        response.raise_for_status()
        return response.content
    except (Timeout, ReadTimeout, ChunkedEncodingError) as e:
        logger.error(f"Error fetching document: {e}")
        raise
    except Exception as e:
        logger.error(e)


def guess_extension(content):
    mimetype = magic.from_buffer(content, mime=True)
    extension = mimetypes.guess_extension(mimetype)
    return extension.replace('*.', '') if extension else ''


def get_text(document_content):
    # simplification of create_inference_batch_object() from demokratibasen
    extension = guess_extension(document_content)
    extracted_content = None
    if extension == ".pdf":
        extracted_content = PdfExtraction(document_content).to_text()
    elif extension == ".docx":
        extracted_content = DocxExtraction(document_content).parse()
    else:
        logger.info(f"Unsupported file extension: {extension}")
        return None
    return extracted_content

def clean_text(text):
    # normalise Unicode to NFKC form
    text = unicodedata.normalize('NFKC', text)
    # optionally, remove non-ASCII characters
    # text = text.encode('ascii', 'ignore').decode('ascii')
    # or, replace ambiguous whitespace with a regular space
    text = re.sub(r'\s+', ' ', text)
    return text

if __name__ == "__main__":
    
    READY = False
    
    logger = get_logger("urls_to_texts")
    documents = get_documents(INPUT_FILE)
    for document in documents: 
        url = document.get("url")
        if not url:
            logger.warning(f"No URL found in document {document['dokument_id']}, skipping.")
            continue
        
        if not READY and url != 'https://innsyn.tromso.kommune.no/application/getMoteDokument?dokid=2001704911':
            continue
        READY = True
        
        document_content = fetch_document(logger, url)
        if not document_content:
            logger.warning(f"No document fetched from {url}.")
            continue
        extracted_text = get_text(document_content)
        if not extracted_text:
            logger.warning(f"No text extracted from document at {url}.")
            continue
        document['tekst'] = clean_text(extracted_text)
        json_data = json.dumps(document, ensure_ascii=False)
        with open(OUTPUT_FILE, "a", encoding="utf-8") as output_file:
            output_file.write(json_data + "\n")
    
