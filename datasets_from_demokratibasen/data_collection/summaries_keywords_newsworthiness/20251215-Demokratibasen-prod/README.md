## Extraction process

* `url-oppsummering-from-prod20251215.csv` contains document data, including summaries, extracted from Demokratibasen
* `batch-files-20251215/` contains files extracted from OpenAI's batch API
* `extract_csv_from_batch_files.py` reads

  * summaries from `url-oppsummering-from-prod20251215.csv` and
  * texts from `batch-files-20251215/input_files/`
* ...and outputs `28081-url-tekst-oppsummering-20251215.csv` and `.pkl`
