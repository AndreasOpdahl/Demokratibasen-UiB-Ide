## Extraction process

In folder `.../<date>-Demokratibasen-prod`:

### Step 1. 
Run
```
./collect_document_and_inferences_from_prod.sh
```
it generates 
* `url-oppsummering-from-prod<date>.csv`, which contains document data, including summaries, extracted from Demokratibasen

### Step 2. 
Run 
```
./download_OpenAI_batch_data.sh
```
it creates and populates
* `batch-files-<date>/`, which contains files extracted from OpenAI's batch API

### Step 3. 
Run
```
python ./extract_csv_from_batch_files.py
```
it reads
* summaries from `url-oppsummering-from-prod<date>.csv` and
* texts from `batch-files-<date>/input_files/`

and generates 
* `<numlines>-url-tekst-oppsummering-<date>.csv` and `.pkl`, which contain (text, summary)-pairs

