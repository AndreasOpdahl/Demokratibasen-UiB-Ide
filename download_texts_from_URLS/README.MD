# Attempt to clear up the folder

Folder should be renamed to `training_data_from_Demokratibasen` or similar.

These are the official attribute names:
dok_id, kommune, url, dok_type, dok_tittel, text, model, max_tokens, oppsum_tittel, oppsummering, personer, nokkelord, nyhetsverdi.

dokument_id,doc_type,kommune,tittel,url,text

Mappings:
dokument.dokument_id/inferens.dokument_id: dok_id
dokument.kommune: kommune
dokument.url: url
dokument.doc_type: dok_type
dokument.tittel: dok_tittel
text: text  # SHOULD BE tekst
model  # SHOULD BE modell
max_tokens
inferens.tittel: oppsum_tittel
inferens.oppsummering: oppsummering
inferens.personer: personer
inferens.nokkelord: nokkelord

## Current files

### Successful training data collection 2025-09-30 (17220 examples)

* `join-44118-summaries-and-15099-texts-from-20250930.py`: joins two datasets in `sources/` folder: `44118-url-tekst-oppsummering-20250930.csv` and `OpenAI_batch_files/15099-texts-from-OpenAI-batch-files-20250930/input_files`. The outputs are `training_data/17720-examples-from-prod-20250930.{csv,pkl,zip,md}`.
* `sources/44118-url-oppsummering-20250930.csv` (and `44118-url-tekst-oppsummering-20250930.pkl`): 44118 document descriptions with dok_id,kommune,dok_type,dok_tittel,url,oppsum_tittel,oppsummering,personer,nokkelord,nyhetsverdi, resulting from a join of the `dokument` and `inferens` tables in Demokratibasen. Generated 2025-09-30.
* `15099-texts-from-OpenAI-batch-files-20250930.jsonl`, downloaded from OpenAI.
* `training_data/17720-examples-from-prod-20250930.{csv,pkl,zip,md}`: 17720 examples from Demokratibasen prod with full texts salvaged from OpenAI batch files. dok_id,kommune,url,dok_type,dok_tittel,text,model,max_tokens,oppsum_tittel,oppsummering,personer,nokkelord,nyhetsverdi. (See the `.md` file for details.)

### Successful training data collection 2025-10-26 (27725 examples)

* `extract_data_from_prod/50301_url_oppsummering_from_prod_20251026.csv`: 50301 document descriptions from Demokratibasen prod with dok_id,kommune,dok_type,dok_tittel,url,oppsum_tittel,oppsummering,personer,nokkelord,nyhetsverdi, extracted as described in `README.md` below.
* `extract_data_from_prod/README.md`: commands for extracting document descriptions from Demokratibasen prod. (`sources/OpenAI_batch_files/README.md` may be more up-to-date-)
* `sources/OpenAI_batch_files/batch-files-20251026/`: contins input and output batch files salvaged from OpenAI 2025-10-26. The batch `input_files/` in this folder were input to `sources/OpenAI_batch_files/extract_csv_from_batch_files.py`.
* `sources/OpenAI_batch_files/extract_csv_from_batch_files.py`: script that inputs document descriptions from `extract_data_from_prod/50301_url_oppsummering_from_prod_20251026.csv` and full texts from
  `sources/OpenAI_batch_files/batch-files-20251026/input_files` and outputs `download_texts_from_URLS/sources/OpenAI_batch_files/27725-url-tekst-oppsummering-20251026.{csv,pkl}` (also stored under `training_data`).
* `training_data/27725-url-tekst-oppsummering-20251026.{csv,pkl}`: 27725 examples from Demokratibasen prod with full texts salvaged from OpenAI batch files.
  dok_id,kommune,url,dok_type,dok_tittel,text,model,max_tokens,oppsum_tittel,oppsummering,personer,nokkelord,nyhetsverdi.

### Other files - perhaps less important

* `log_entries\`
* `sources.txt`
* `urls_with_missing_doctext.csv`
* `urls.txt`

## This is the GitHub catalogue structure:

```
    download_texts_from_URLS/
        dokument_jsons/
        logs/
        training_data/
        en-20250624.csv
        database.py
        join-44118-summaries-and-15099-texts-from-20250930.py
        log_entries
        pdfextraction.py
        sources.txt
        split_jsonl.py
        training_data.db
        urls.txt
        urls_to_texts.py
        urls_with_missing_doctext.csv
```
