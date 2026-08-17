## Extraction process

Run in this folder to create and populate`.../<date>-Demokratibasen-prod`:

### Step 1.

Run

```
./collect_document_and_inferences_from_prod.sh
```

it generates

* `./<date>-Demokratibasen-prod/url-oppsummering-from-prod<date>.csv`, which contains document data, including summaries, extracted from Demokratibasen

### Step 2.

Run

```
./download_OpenAI_batch_data.sh
```

it creates and populates

* `./<date>-Demokratibasen-prod/batch-files-<date>/`, which contains files extracted from OpenAI's batch API

### Step 3.

Run

```
python ./join_texts_and_summaries.py
```

it reads

* summaries from `./<date>-Demokratibasen-prod/url-oppsummering-from-prod<date>.csv` and
* texts from `./<date>-Demokratibasen-prod/batch-files-<date>/input_files/`

and generates

* `./<date>-Demokratibasen-prod/<numlines>-url-tekst-oppsummering-<date>.csv` and `.pkl`, which contain the discovered new (text, summary)-pairs
