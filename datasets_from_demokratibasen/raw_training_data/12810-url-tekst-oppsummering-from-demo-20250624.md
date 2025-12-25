## Lineage of 12810-url-tekst-oppsummering-20250624.csv

Inputs:1. 36812-demokratibasen-texts-20250528.jsonl — document texts (derived from Demokratibasen-demo dokument table dump, texts downloaded via urls_to_texts.py in May 2025)

1. 17569-demokratibasen-inferences-20250624.csv — GPT-3.5-turbo inferences (dump from Demokratibasen-demo inferens table, June 2025)

Process:* Natural join on dokument_id

* Filtered to 4 doc_types: meeting_agenda, meeting_minutes, case_presentation, case_minutes
* Columns aligned to match 27725-url-tekst-oppsummering-20251026.csv schema
* model set to gpt-3.5-turbo

Output: 12,810 rows with text + summaries/keywords/newsworthiness
