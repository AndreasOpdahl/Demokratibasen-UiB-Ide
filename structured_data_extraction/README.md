Extracts people, places, organisations, events, times, themes, etc from documents.

The purpose is to generate training data for fine-tuning data extractors.

Uses a prompt similar to summary_generation.

## Fields

These are the official attribute names:
dok_id, komm_nr, url, dok_type, dok_tittel, tekst, modell, maks_tokens, oppsum_tittel, oppsummering, personer, nokkelord, nyhetsverdi.

dokument_id,doc_type,kommune,tittel,url,text

Mappings:
dokument.dokument_id/inferens.dokument_id: dok_id
dokument.kommune: kommune
dokument.url: url
dokument.doc_type: dok_type
dokument.tittel: dok_tittel
text: tekst
model: modell
max_tokens: maks_tokens
inferens.tittel: oppsum_tittel
inferens.oppsummering: oppsummering
inferens.personer: personer
inferens.nokkelord: nokkelord

## Results files

* `extracted_data_MODEL.jsonl` contains analyses of full INITIAL datasets using BASIC prompt. MODELS: claude-3-sonnet, gemini, openai, ...
