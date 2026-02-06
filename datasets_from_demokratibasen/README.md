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

