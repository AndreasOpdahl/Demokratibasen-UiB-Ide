Scripts for extracting training data from demokratibasen-prod and from OpenAI's batch API.

* The CSV files are SQL dumps from running Demokratibasen instances.
* The JSONL files are texts scraped from public PDF and DOCX documents. They are used to create training data, for examples pairs of
 ```
 <public_document_texts, document_summaries>
 ```

File names:
```
 <num_of_examples>-`<instance-name>`-{urls,texts,inferences}-<extraction_date>.{csv,jsonl}
```

# Collected analysis results (without raw texts)

../training_data: 17720 examples from demokratibasen-prod on 2025-09-30 and -31.


# Collected raw texts

???


## To collect analysis results from demokratibasen-prod

Run
```
. ./collect_documents_and_inferences_from_prod.sh
```

## Alternative stepwise procedure (OLD)

ssh demokratibasen-uib-ide

On host:
docker exec -ti demokratibasen-demo-db-1 bash

In docker:
psql postgres -U postgres
\copy (SELECT dokument_id,doc_type,kommune,tittel,url FROM dokument) TO 'demokratibasen-uib-ide-urls-20250920.csv' WITH CSV HEADER;
% variation: include doc_tekst!
exit

Back in host:
docker cp demokratibasen-demo-db-1:demokratibasen-uib-ide-urls-20250920.csv .
exit

Back on local:
scp demokratibasen-uib-ide:demokratibasen-uib-ide-urls-20250920.csv .

To extract inferences:

\copy (SELECT dokument_id,batch_id,tittel,oppsummering,personer,nokkelord,nyhetsverdi FROM inferens) TO 'demokratibasen-uib-ide-inferences-20250920.csv' WITH CSV HEADER;

## To collect analysis results from demokratibasen-prod

The full texts are retrieved from OpenAI batch files. See the sources-subfolder for COMMANDS.txt.

## To scrape texts

The script urls_to_texts.py .
