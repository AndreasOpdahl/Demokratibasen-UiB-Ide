Files/Folders with Document Text

#	Type	Path	Text Count	Unique IDs	ID Field	Fields

Identical Sets: #1 (36812-demokratibasen-texts-20250528.jsonl)	#7 (dokument_jsons/)	35,133
1	JSONL	.../202505-and-06-Demokratibasen-demo/36812-demokratibasen-texts-20250528.jsonl	35,134	35,133	dokument_id	dokument_id, doc_type, kommune, tittel, url, tekst
7	JSON_DIR	.../202505-and-06-Demokratibasen-demo/dokument_jsons	35,133	35,133	dokument_id	dokument_id, doc_type, kommune, tittel, url, tekst

-- DISJOINT ---

Identical Sets: #3 (27725-url-tekst-oppsummering-20251026.csv)	#6 (raw_training_data/27725-url-tekst-oppsummering-20251026.csv)	27,688
3	CSV	.../20251026-Demokratibasen-prod/27725-url-tekst-oppsummering-20251026.csv	27,688	27,688	dok_id	dok_id, kommune, url, dok_type, dok_tittel, text, model, max_tokens, oppsum_tittel, oppsummering, personer, nokkelord, nyhetsverdi
6	CSV	raw_training_data/27725-url-tekst-oppsummering-20251026.csv	27,688	27,688	dok_id	(same as #3)

Overlapping Sets: #3 (27,688)	#5 (14,331)	10,983
Overlapping Sets: #5 (14,331)	#6 (27,688)	10,983
5	CSV	raw_training_data/17720-examples-from-prod-20250930.csv	17,710	14,331	dok_id	(same as #3)

Overlapping Sets: #2 (9,549)	#6 (27,688)	2,590
Overlapping Sets: #2 (9,549)	#3 (27,688)	2,590
Overlapping Sets: #2 (9,549)	#5 (14,331)	2,069
2	JSONL	.../20250920-21-misc-Demokratibasen-dumps/103908-dokumenter-texts-20250921.jsonl	9,549	9,549	dokument_id	dokument_id, url, tekst

4	CSV	.../20251125-Demokratibasen-prod/1584-url-tekst-oppsummering-20251125.csv	1,584	1,584	dok_id	(same as #3)

Other Overlapping Sets
Dataset A	Dataset B	Common IDs
#1 (35,133)	#5 (14,331)	673
#5 (14,331)	#7 (35,133)	673
#3 (27,688)	#4 (1,584)	40
#4 (1,584)	#6 (27,688)	40
#2 (9,549)	#4 (1,584)	16
#4 (1,584)	#5 (14,331)	25
