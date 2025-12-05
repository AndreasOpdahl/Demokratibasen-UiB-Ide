# Successful training data collection 2025-10-26 (27725 examples)

## Files

* `extract_data_from_prod/50301_url_oppsummering_from_prod_20251026.csv`: 50301 document descriptions from Demokratibasen prod with dok_id,kommune,dok_type,dok_tittel,url,oppsum_tittel,oppsummering,personer,nokkelord,nyhetsverdi, extracted as described in `README.md` below.
* `extract_data_from_prod/README.md`: commands for extracting document descriptions from Demokratibasen prod. (`sources/OpenAI_batch_files/README.md` may be more up-to-date-)
* `sources/OpenAI_batch_files/batch-files-20251026/`: contins input and output batch files salvaged from OpenAI 2025-10-26. The batch `input_files/` in this folder were input to `sources/OpenAI_batch_files/extract_csv_from_batch_files.py`.
* `sources/OpenAI_batch_files/extract_csv_from_batch_files.py`: script that inputs document descriptions from `extract_data_from_prod/50301_url_oppsummering_from_prod_20251026.csv` and full texts from
  `sources/OpenAI_batch_files/batch-files-20251026/input_files` and outputs `download_texts_from_URLS/sources/OpenAI_batch_files/27725-url-tekst-oppsummering-20251026.{csv,pkl}` (also stored under `training_data`).
* `training_data/27725-url-tekst-oppsummering-20251026.{csv,pkl}`: 27725 examples from Demokratibasen prod with full texts salvaged from OpenAI batch files.
  dok_id,kommune,url,dok_type,dok_tittel,text,model,max_tokens,oppsum_tittel,oppsummering,personer,nokkelord,nyhetsverdi.
