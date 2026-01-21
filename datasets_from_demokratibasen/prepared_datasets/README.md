
## Origins

### OLD gpt-3.5-turbo dataset

* `OLD_text_summary_dataset_12811_examples_gpt_35_turbo/` 12811 examples. Corresponds to `../raw_training_data/` from 20250624, sourced from Demokratibasen-demo with GPT-3.5-turbo summaries
  * This was the first dataset used in the first weeks of the UiB-Idé project

### gpt-4o-mini

* `text_summary_dataset_202505`: 13077 examples. Corresponds to `../raw_training_data/` from 20250624, sourced from Demokratibasen-demo, but with new gpt-4o-mini summaries
* `text_summary_dataset_bergen_2017_2023`: 121466 examples. Contains documents collected from Bergen municipality from 2017-2023 (and some earlier?), with new gpt-4o-mini summaries
  * documents were collected by a separate project `Kommunebasen-Bergen`
  * the file `.../dataset_bergen_2017_2023/dokumenter.jsonl` contains (many of - but apparently not all) the original documents
* `text_summary_dataset_202505_to_12/`: 47215 examples. Merger of
  * `text_summary_dataset_202505_to_10/` and
  * `text_summary_dataset_202511_and_12/`
* `text_summary_dataset_202505_to_10/`: 43221 examples. Corresponds to
  * This was the second dataset used in the autumn/winter of 2025
  * `../raw_training_data/` from 20250624, 20250930, and 20251026, sourced from Demokratibasen-prod and OpenAI batches with GPT-4o-mini summaries
  * includes `OLD_text_summary_dataset_12811_examples_gpt_35_turbo/`, so 12811 of the summaries are GPT-3.5-turbo
  * lots of overlap with `text_summary_dataset_29665_examples/`
* `text_summary_dataset_202511_and_12/`: 28764 (was 29665) examples. Corresponds to
  * `../raw_training_data/` from 20251125 and 20251215, sourced from Demokratibasen-prod and OpenAI batches with GPT-4o-mini summaries
