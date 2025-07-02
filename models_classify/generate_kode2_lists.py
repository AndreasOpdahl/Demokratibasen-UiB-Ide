from summarize_labelling import sample_csv, return_document_dict
import pandas as pd
from structured_documenttype_class import classify
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


DOCUMENTS_CSV_PATH = "./download_texts_from_URLS/dokumenter.csv"
"""
df = pd.read_csv(DOCUMENTS_CSV_PATH)
print(df.head())

doc_id = df["dokument_id"].iloc[0]
kommune_nr = df["kommune"].iloc[0]
print(doc_id)
doc_dict = return_document_dict(doc_id, kommune_nr)
print(doc_dict["tekst"])

print(df.info())
"""
# Chose random state as 45 as it had 499 valid sample elements that existed in the dokument_jsons folder
sample_df = sample_csv(DOCUMENTS_CSV_PATH, "./models_classify/sample_documents.csv", n = 500, random_state= 45)
sample_list = [{"dokument_id": sample_df["dokument_id"].iloc[i], "kommune": sample_df["kommune"].iloc[i]} for i in range(len(sample_df))]

#print(sample_list[0])
def create_kode2(provider, dict_list):
    kode2_list = []
    i=0
    for dict_ele in dict_list:
        dok_id = dict_ele["dokument_id"]
        kom_id = dict_ele["kommune"]
        try:
            doc_dict = return_document_dict(dok_id, kom_id)
        except:
            continue
        tittel = doc_dict["tittel"]
        tekst = doc_dict["tekst"]
        try:
            model_response_str = classify(tittel, tekst, "kode2", provider)
            print(f"Classified document #{i}")
            model_response_json = json.loads(model_response_str)
            kode2 = model_response_json["kode2"]
            begrunnelse = model_response_json["begrunnelse_kode2"]

            kode2_list.append({"kode":kode2, "begrunnelse":begrunnelse})
            i += 1
        except:
            continue

        
    return kode2_list
"""
if __name__ == "__main__":
    gpt_kode2 = create_kode2("openai", sample_list)
    gemini_kode2 = create_kode2("gemini", sample_list)
    claude_kode2 = create_kode2("claude", sample_list)
    kode2_lists ={
        "openai": gpt_kode2,
        "gemini": gemini_kode2,
        "claude": claude_kode2
    }
    with open("kode2.json", "w") as outfile:
        json.dump(kode2_lists, outfile)
"""
if __name__ == "__main__":
    providers = ["openai", "gemini", "claude"]
    kode2_lists = {}

    # spin up a pool with one thread per provider
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        # submit one job per provider
        futures = {
            executor.submit(create_kode2, provider, sample_list): provider
            for provider in providers
        }

        # as each finishes, grab the result and store it
        for future in as_completed(futures):
            provider = futures[future]
            try:
                kode2_lists[provider] = future.result()
                print(f"{provider} done, {len(kode2_lists[provider])} items")
            except Exception as e:
                print(f"{provider} raised an exception: {e!r}")
                kode2_lists[provider] = []

    # write all three into one JSON
    with open("kode2.json", "w") as outfile:
        json.dump(kode2_lists, outfile, ensure_ascii=False, indent=2)
