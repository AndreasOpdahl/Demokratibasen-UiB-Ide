## Contents

17720 training examples with dok_id, dok_type, url, tittel, tekst, modell, max_tokens, oppsumm_tittel, oppsummering, navn, nokkelord, tema, nyhetsverdi.

Extracted from demokratibasen-prod 2025-09-30 and OpenAI batch files ("30 last days") with full texts.

THE COMMANDS BELOW HAVE BEEN SUPERCEDED. 

See the README.MD file in the download_texts_from_URLS/ folder (which may since have been renamed).

## To extract from demokratibasen-prod

```
ssh demokratibasen-uib-ide
```

On host:

```
docker exec -ti demokratibasen-demo-db-1 bash
```

In docker:

```
psql postgres -U postgres

\COPY (SELECT d.dokument_id as dok_id, d.kommune as kommune, d.doc_type as dok_type, d.tittel as dok_tittel, d.url as url, i.tittel as oppsum_tittel, i.oppsummering as oppsummering, i.personer as personer, i.nokkelord as nokkelord, i.nyhetsverdi as nyhetsverdi FROM inferens i JOIN dokument d ON i.dokument_id = d.dokument_id) TO 'csv_fil.csv' WITH CSV HEADER;

exit
```

Back in host:

```
docker cp demokratibasen-demo-db-1:csv_fil.csv .

exit
```

Back on local:

```
scp demokratibasen-uib-ide:csv_fil.csv .
```

## To retrieve batch files with full text

Retrieve all files (batch files from last 30 days):

```
curl https://api.openai.com/v1/files -H "Authorization: ${OPENAI_API_KEY}"   -H "Content-Type: application/json" > file_list
```

Pick batch input files

```
jq ".data[] | select(.purpose == \"batch\") | .id" file_list | sed "s/\"//g" > batch_file_list
```

Retrieve file context

```
for file_id in $(cat batch_file_ids); do curl https://api.openai.com/v1/files/${file_id}/content -H "Authorization: Bearer ${OPENAI_API_KEY}" -H "Content-Type: application/json" > ${file_id}.json; done
```

Further processing:

```
./download_texts_from_URLS/join-44118-summaries-and-15099-texts-from-20250930.py
```
