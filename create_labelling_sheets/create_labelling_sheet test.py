import json
import pandas as pd
import random

INPUT_FILE = "dokumenter.jsonl"
OUTPUT_FILE = "labelling_sheet.csv"

NUMBER_OF_DOCS_TO_LABEL = 110

document_types_to_consider = [
    # these are all the document types already used in Demokratibasen
    "meeting_agenda",
    "meeting_minutes",
    "case_presentation",
    "case_minutes",
    "case_attachment",
    "case_history",
]
# how many documents, relatively, of each type to include in the labelling sheet
document_type_ratio = {
    "meeting_agenda": 1,
    "meeting_minutes": 1,
    "case_presentation": 5,
    "case_minutes": 1,
    "case_attachment": 5,
    "case_history": 5,
}


# read and shufle the documents
with open(INPUT_FILE, "r", encoding="utf-8") as innfil:
    dokumenter = [
        json.loads(linje)
        for linje in innfil
    ]
print(f"Read {len(dokumenter)} documents from {INPUT_FILE}")
random.shuffle(dokumenter)

# set target numbers of documents per type
sum_ratios = sum(document_type_ratio.values())
document_type_target = {
    doctype: int((ratio / sum_ratios) * NUMBER_OF_DOCS_TO_LABEL)
    for doctype, ratio in document_type_ratio.items()
}

valgte_dokumenter = []
for dok in dokumenter:
    doktype = dok.get('doc_type')
    if document_type_target.get(doktype, 0) <= 0:
        continue
    valgte_dokumenter.append(dok)
    document_type_target[doktype] -= 1
random.shuffle(valgte_dokumenter)

sheet_data = [
    {
        'kode': '',
        'tittel': dok['tittel'][:200] if dok.get('tittel') else '',
        'korttekst': dok['tekst'][:500] if dok.get('tekst') else '',
        'fulltekst': dok['tekst'] if dok.get('tekst') else '',
        'kommune': dok.get('kommune', ''),
        'dokument_id': dok.get('dokument_id', '')
    }
    for dok in valgte_dokumenter
]
sheet_df = pd.DataFrame(sheet_data)
sheet_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
