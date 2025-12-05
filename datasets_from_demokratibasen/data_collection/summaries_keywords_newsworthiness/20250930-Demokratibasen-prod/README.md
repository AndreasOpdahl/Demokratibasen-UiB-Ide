# Successful training data collection 2025-09-30 (17220 examples)

## Files

* `join-44118-summaries-and-15099-texts-from-20250930.py`: joins two datasets in `sources/` folder: `44118-url-tekst-oppsummering-20250930.csv` and `OpenAI_batch_files/15099-texts-from-OpenAI-batch-files-20250930/input_files`. The outputs are `training_data/17720-examples-from-prod-20250930.{csv,pkl,zip,md}`.
* `44118-url-oppsummering-20250930.csv` (and `44118-url-tekst-oppsummering-20250930.pkl`): 44118 document descriptions with dok_id,kommune,dok_type,dok_tittel,url,oppsum_tittel,oppsummering,personer,nokkelord,nyhetsverdi, resulting from a join of the `dokument` and `inferens` tables in Demokratibasen. Generated 2025-09-30.
* `15099-texts-from-OpenAI-batch-files-20250930.jsonl`, downloaded from OpenAI.
* ../`training_data/17720-examples-from-prod-20250930.{csv,pkl,zip,md}`: 17720 examples from Demokratibasen prod with full texts salvaged from OpenAI batch files. dok_id,kommune,url,dok_type,dok_tittel,text,model,max_tokens,oppsum_tittel,oppsummering,personer,nokkelord,nyhetsverdi. (See the `.md` file for details.)
