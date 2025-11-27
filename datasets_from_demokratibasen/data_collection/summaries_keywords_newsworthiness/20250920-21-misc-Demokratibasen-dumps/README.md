#### Bulk download from various demokratibasen-* instances (old -demo, new -prod, UiB-Idé -test)

'''The ''big'' problem here is that we may miss texts for many of them...'''

* `12243-demokratibasen-uib-ide-texts-20250920.csv`: 12 243 rows with columns dokument_id, doc_type, kommune, tittel, url, doc_tekst. Generated 2025-09-20. Same fields as the corresponding `...-urls-....csv` file but with doc_tekst too.
* `12243-demokratibasen-uib-ide-urls-20250920.csv`: 12 243 rows with columns dokument_id, doc_type, kommune, tittel, url (dump of the `dokument` table). Generated 2025-09-20.
* `111721-demokratibasen-test-urls-20250920.csv`: 111 721 rows with columns dokument_id, doc_type, kommune, tittel, url (dump of the `dokument` table). Generated 2025-09-20.
* `29602-demokratibasen-test-inferences-20250920.csv`: 29 602 rows with columns dokument_id, batch_id, tittel, oppsummering, personer, nokkelord, nyhetsverdi (dump of the `inferens` table). Generated 2025-09-20.
* `103908-dokumenter-texts-20250921.jsonl`: 9 609 objects with keys dokument_id, tekst, url. Generated 2025-09-21.(This is a `.jsonl` file. Where did it come from?)
* `111188-demokratibasen-prod-urls-20250921.csv`: 111 188 rows with columns dokument_id, doc_type, kommune, tittel, url (dump of the `dokument` table). Generated 2025-09-21.
* `29281-demokratibasen-prod-inferences-20250921.csv`: 29 281 rows with columns dokument_id, batch_id, tittel, oppsummering, personer, nokkelord, nyhetsverdi (dump of the `inferens` table). Generated 2025-09-21.
* `6100-demokratibasen-uib-ide-inferences-20250920.csv`: 6 100 rows with columns dokument_id, batch_id, tittel, oppsummering, personer, nokkelord, nyhetsverdi (dump of the `dokument` table). Generated 2025-09-20.
