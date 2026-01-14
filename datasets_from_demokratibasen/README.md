# Datasets extracted from Demokratibasen

## Folders

* `data_collection/` : Scripts used for data collection and intermediate data that were generated.
* `raw_training_data/` : The outputs from `data_collection/` but not in standardised `dataset/` format.
  * These are flat document-level datasets (dokument_id + text + basic metadata).
  * See the "Attribute names" below.
* `prepared_datasets/` : Processed and aggregated raw-training data files: they wrap text into input, labels into output, and move/enrich document metadata into a nested metadata dict (with additional fields such as personer, nokkelord, nyhetsverdi that are not present in the raw file).
* `datasets/` : The processed and aggregated datasets to use for training purposes.
  * The 12811 dataset is legacy. All the documents are included in the larger 43221 dataset.
  * The larger 43221 dataset covers all dok_id values from the CSV files:
    * `raw_training_data/`17720-examples-from-prod-20250930.csv
    * `raw_training_data/`27725-url-tekst-oppsummering-20251026.csv
  * `test_summary_dataset_ALL_examples`: this is the largest and most recent dataset with 160 000+ examples

# Attribute names

These are the official attribute names:

**dok_id, kommune, url, dok_type, dok_tittel,** *text ->* **tekst,** *model ->* **modell,** *max_tokens ->* **maks_tokens, oppsum_tittel, oppsummering, personer, nokkelord, nyhetsverdi.**

dokument_id,doc_type,kommune,tittel,url,text

Mappings:
dokument.dokument_id/inferens.dokument_id: dok_id
dokument.kommune: kommune
dokument.url: url
dokument.doc_type: dok_type
dokument.tittel: dok_tittel
text: text  # SHOULD BE tekst
model  # SHOULD BE modell
max_tokens  # SHOULD BE maks_tokens
inferens.tittel: oppsum_tittel
inferens.oppsummering: oppsummering
inferens.personer: personer
inferens.nokkelord: nokkelord
