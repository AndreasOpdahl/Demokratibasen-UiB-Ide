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
