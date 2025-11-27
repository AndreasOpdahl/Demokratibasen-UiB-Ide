# The first data collection May-June 2025

* Period: May-June 2025
* Data source: most likely Demokratibasen-demo
* Model: most likely gpt-3.5-turbo
* Number of documents: ??
* Attributes: dokument_id,doc_type,kommune,tittel,url,text; dokument_id, batch_id, oppsumering_tittel, oppsummering, personer, nokkelord, nyhetsverdi; ???
* File: AJAY cleaned and merged these files -> it was his first dataset!

## Files

* `36812-demokratibasen-urls-20250528.csv`: 41 706 document descriptions with columns dokument_id, doc_type, kommune, tittel, url. (This is a dump of the `dokument` table in Demokratibasen. Perhaps many of them had document types that were not analysed by ChatGPT.) Generated 2025-05-28.
* `urls_to_texts.py`, `pdf_extraction.py` and `split_jsonl.py`: scripts used to download `dokument_jsons` full texts based on the document descriptions in `36812-demokratibasen-urls-20250528.csv` from 2025-06-16 or earlier.
* `logs/`: `urls_to_texts` logs from 2025-05-27, when the texts in `dokument_jsons/` were doenloaded. Also logs from a run 2025-09-30 (I think the latter one failed because the scrapers had been rewritten and some municipalities had switched systems, therefore the full texts were salvaged from OpenAI batch input files instead).
* `dokument_jsons/`: 36812 full texts, with dokument_id,doc_type,kommune,tittel,url,text from 2025-06-16 or earlier. Municipalities: 4601|5501|5536.
* `36812-demokratibasen-texts-20250528.jsonl`: 36 813 objects with keys doc_type, dokument_id, kommune, tekst, tittel, url. Generated 2025-05-28. Same fields as the corresponding `...-urls-....csv` file but with doc_tekst too. Compiled from `dokument_jsons/` with the `split_jsonl.py` script.
* `17569-demokratibasen-inferences-20250624.csv`: 16 276 rows with columns dokument_id, batch_id, tittel, oppsummering, personer, nokkelord, nyhetsverdi. (This is a dump of the `inferens` table in Demokratibasen. It is possible that some dokument_id-er were analysed several times, perhaps with different GPT models.) Generated 2025-06-24.
