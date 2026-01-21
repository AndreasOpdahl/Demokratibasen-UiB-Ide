# Models Classify README

This directory contains Python files for classifying data using various AI models. To use these files, you need to create a `.env` file in the root of your project to store API keys for the models you want to use.

## Required API Keys

The `.env` file should include the following keys:

- **OPENAI_API_KEY**: API key for OpenAI's models.
- **GOOGLE_API_KEY**: API key for Google's Gemini model.
- **ANTHROPIC_API_KEY**: API key for Anthropic's Claude model.

### Example `.env` File

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Make sure to replace `your_openai_api_key_here`, `your_google_api_key_here`, and `your_anthropic_api_key_here` with your actual API keys.

## Usage

Once the `.env` file is set up, you can run the Python files in this directory to interact with the models. Ensure you have installed all required dependencies and have access to the APIs. The .gitignore file automatically ignores your .env file, so you can still use git and not be worried about sharing your api-keys.

## What has been done
Created a temporarily codeset for kode2. Used Gemini, Openai and Claude for generating codesets and manually inspected these. A lot of testing and experimenting has been done, and the project has still a lot of work to be done.

## Files

### classify_documenttypes.py

File that uses *temporarily* kode2 labels (with descriptions). Has functions that selects random json files from a folder path. The folder path in our code base is the folder that contains json files about the different documents(download_texts_from_URLs/dokument_jsons). extract_json function is only a function that uses regex to find the first text block that starts and ends with "{","}", however if the json file starts with "[" and ends with "]" it doesnt work. trunctate_text_to_fit uses gpt-tokenizer and count tokens and then trunctates the text to fit the number of tokens for a given model. The max amount of tokens is set in MAX_TOKENS for each model. classify_document is the function that uses the prompts set at the start of the file, and returns json response.
worker is the function that each of the threads will use when running. It completes every prompt given from the document_paths list.
The main loop runs every model, and saves the results under the folder model_outputs. It has been ran twice. Make sure if ran more times that you don't overwrite already existing files. Rename the output-file path in lines: 244 and 305 (out_path variable)

### generate_kode2_lists.py
File to generate the list of generating "kode2" for ~500 sample elements. Saved into kode2.json, used for testing

### json_to_csv.py
File that has manually all the codes originally created from the different models. Has been manually picked to remove codes already existing in kode1 and to remove overlapping categories. The file if run through main-loop prints unique values created by the models, and prints a counter-string for each of the created codes.

### structured_documenttype_class.py
File that has some functions that is used in other files including classify() which classifies either kode1 or kode2 for a model. However a more general function should probably be created so it's easier to use different prompts and models. main-loop used to create labelled_with_xmodel where xmodel is either claude, gemini or gpt. This was used for the test and validation set to ensure the models created correct classifications.

### summarize_labelling.py
File that has different functions to use in the other files. These includes count_values() which takes a list and returns a dictionary which has all the values from the lists as keys, and the number of occurences as values.
return_majority() which returns the element from a list which occurs the most.
check_unanimity() checks if all of the values in a list is the same, if it is it returns True, else False.
sample_csv() takes a csv file and samples n amount of rows given a column (and valid values of that column). Standard is "kommune" with the values 4601, 5501, and 5536. These were chosen as these had valid values that had been scraped.

### claude_kode2.txt
text response from claude that has list of codes the given descriptions.

### gemini_kode2.json
json response from gemini that has a list of all of the codes and the given descriptions.

### openai_kode2.json
json response from openai (gpt4o) that has a list of all of the codes and the given descriptions

### kode2.json
kode2 for claude, openai and gemini for the 500 sampled documents.

### kode2_samlet.json
kode2 with explanations, is a code-set of 37 codes, should probably be expanded and alter to be more fitting of the documents.

### sample_documents.csv
A csv file containing ~500 samples with the text and title. Some texts and/or formats are too large for models and should be furthered cleaned and processed.

### labelled_with_claude.csv
Created from structured_documenttype_class.py. Uses the validation set.

### labelled_with_gemini.csv
Created from structured_documenttype_class.py. Uses the validation set.

### labelled_with_gpt.csv
Created from structured_documenttype_class.py. Uses the validation set.

### labelling_sheet_summarized_data_test.csv
Test set created from the original google_sheet under UiB idé.

### labelling_sheet_summarized_data_validation.csv
Validation set created from the original google_sheet under UiB idé.

## Further work
- Create more general functions and implement classes and/or interface-like classes to ensure similar behaviour across models. 
- Further work on the codeset for kode2 both by using LLMs and manual inspection
- Gather the code and systemize it into more logical and intuitive structure. This includes splitting functions into smaller functions and having different files for different use-cases.
- After cleaning code, remove old and bad code. During the summer-project the code was created as-needed and not done in a proper and structured way for further scaling.
- Gather information and use-cases by consulting with experts(journalists, people who initially creates these documents, and people who will use the tools.)

