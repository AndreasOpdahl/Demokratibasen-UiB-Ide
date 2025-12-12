import csv
import pandas as pd
import json 

def count_values(values: list) -> dict:
    results = {}
    for val in values:
        if val not in results:
            results[val] = 1
        else:
            results[val] += 1

    return results

def return_majority(values:list):
    results = count_values(values)
    max_key = max(results, key=results.get)
    max_value = results[max_key]
    min_key = min(results, key=results.get)
    min_value = results[min_key]
    if max_value == min_value and min_value < len(values):
        return "EQUAL"
    else:
        return max_key
    
def check_unanimity(values:list) -> bool:
    last_value = None
    for value in values:
        if last_value == None:
            last_value = value
        if last_value != value:
            return False
    return True

# Function  taking a csv file and 
def sample_csv(input_csv_path: str,
               output_csv_path: str | None = None,
               n: int = 500, 
               random_state: int | None = None, 
               filter_column = "kommune", 
               filter_values = [4601, 5501, 5536]) -> pd.DataFrame:
    # Read the full CSV, and sample it
    df = pd.read_csv(input_csv_path)
    filtered_df = df[df[filter_column].isin(filter_values)]
    sample_df = filtered_df.sample(n=n, random_state=random_state)
    
    if output_csv_path:
        sample_df.to_csv(output_csv_path, index=False)
    
    return sample_df

def return_document_dict(doc_id, kommune_nr):
    path = f"./download_texts_from_URLS/dokument_jsons/{kommune_nr}_{doc_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
#print(count_values([1,3,2]))