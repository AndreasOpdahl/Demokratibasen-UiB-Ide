import json
import csv
import pandas as pd

ORIGINAL_KODE2 = [
"Administrativt_vedtak",
"Artikkel",
"Bilag",
"Byggesak",
"Erklæring",
"Høringsdokument",
"Internt notat",
"Kart",
"Klage",
"Handlingsplan",
"Internkontroll",
"Møteagenda",
"Orientering",
"Planverk",
"Spørsmål og svar",
"Presentasjon",
"Risikoanalyse",
"Saksdokument",
"Strategisk dokument",
"Søknad",
"Tegninger",
"Vedtak",
"Økonomidokument",
"Rammetillatelser",
"Dispensasjoner",
"Digitalisering",
]

def json_to_csv_file(from_path, to_path, *keys):
    with open(from_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)[[key for key in keys]]
    df.to_csv(to_path, index=False, encoding="utf-8")

def get_values_from_key(from_path, key):
    with open(from_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    values = [element[key] for element in data]
    return values

def get_difference_from_lists(list_1, list_2):
    set_1 = set(list_1)
    set_2 = set(list_2)
    return set_1.difference(set_2)

def get_intersection_from_lists(list_1, list_2):
    set_1 = set(list_1)
    set_2 = set(list_2)
    return set_1.intersection(set_2)

def generate_counter_dict(from_path):
    counter_dict = {}
    with open(from_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if item["kode2"] in counter_dict:
            counter_dict[item["kode2"]] += 1
        else:
            counter_dict[item["kode2"]] = 1
    return counter_dict


if __name__ == "__main__":
    #json_to_csv_file("./model_outputs/kode2_claude_2.json", "./model_outputs/csv_kode2_claude_2.csv", "tittel", "tekst", "kode2", "begrunnelse_kode2")
    #json_to_csv_file("./model_outputs/kode2_openai_2.json", "./model_outputs/csv_kode2_openai_2.csv", "tittel", "tekst", "kode2", "begrunnelse_kode2")
    #json_to_csv_file("./model_outputs/kode2_gemini_2.json", "./model_outputs/csv_kode2_gemini_2.csv", "tittel", "tekst", "kode2", "begrunnelse_kode2")

    claude = get_values_from_key("./model_outputs/kode2_claude_2.json", "kode2")
    openai = get_values_from_key("./model_outputs/kode2_openai_2.json", "kode2")
    gemini = get_values_from_key("./model_outputs/kode2_gemini_2.json", "kode2")
    print(f"Unique original kode2 values: {len(ORIGINAL_KODE2)}")
    print(f"Unique kode2 values for claude: {len(set(claude))}")
    print(f"Unique kode2 values for openai: {len(set(openai))}")
    print(f"Unique kode2 values for gemini: {len(set(gemini))}")
    print()
    print(f"Unique kode2 values for claude not in the original: {len(get_difference_from_lists(claude, ORIGINAL_KODE2))}")
    print(f"Unique kode2 values for openai not in the original: {len(get_difference_from_lists(openai, ORIGINAL_KODE2))}")
    print(f"Unique kode2 values for gemini not in the original: {len(get_difference_from_lists(gemini, ORIGINAL_KODE2))}")
    print()
    print(f"Unique kode2 values used in original but not claude: {len(get_difference_from_lists(ORIGINAL_KODE2, claude))}")
    print(f"Unique kode2 values used in original but not openai: {len(get_difference_from_lists(ORIGINAL_KODE2, openai))}")
    print(f"Unique kode2 values used in original but not gemini: {len(get_difference_from_lists(ORIGINAL_KODE2, gemini))}")

    print(f"List of unique kode2 values used in original but not claude: {get_difference_from_lists(ORIGINAL_KODE2, claude)}")
    print(f"List of unique kode2 values used in original but not openai: {get_difference_from_lists(ORIGINAL_KODE2, openai)}")
    print(f"List of unique kode2 values used in original but not gemini: {get_difference_from_lists(ORIGINAL_KODE2, gemini)}")

    print(f"List of unique kode2 values used in claude but not original: {get_difference_from_lists(claude, ORIGINAL_KODE2)}")
    print(f"List of unique kode2 values used in openai but not original: {get_difference_from_lists(openai, ORIGINAL_KODE2)}")
    print(f"List of unique kode2 values used in gemini but not original: {get_difference_from_lists(gemini, ORIGINAL_KODE2)}")

    print("Claude:")
    print(generate_counter_dict("./model_outputs/kode2_claude_2.json"))
    print("OpenAI:")
    print(generate_counter_dict("./model_outputs/kode2_openai_2.json"))
    print("Gemini")
    print(generate_counter_dict("./model_outputs/kode2_gemini_2.json"))
    print("Claude 2:")
    print(generate_counter_dict("./model_outputs/kode2_claude_2.json"))
    print("OpenAI 2:")
    print(generate_counter_dict("./model_outputs/kode2_openai_2.json"))
    print("Gemini 2:")
    print(generate_counter_dict("./model_outputs/kode2_gemini_2.json"))