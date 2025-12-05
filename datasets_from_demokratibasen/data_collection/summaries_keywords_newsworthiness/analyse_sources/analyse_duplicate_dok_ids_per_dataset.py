"""Analyses duplicate dok_id values in the canonical datasets."""


from analyse import canonical_data, canonical_joined_examples, canonical_json_texts


def duplicate_dok_ids(rows):
    """Return the set of dok_id values that appear more than once in rows."""
    counts = {}
    for row in rows:
        dok_id = row.get("dok_id")
        if dok_id is None:
            continue
        counts[dok_id] = counts.get(dok_id, 0) + 1
    return {dok_id for dok_id, count in counts.items() if count > 1}


def differing_keys(dicts):
    """Return the set of keys whose values differ across the given dicts."""
    if not dicts:
        return set()
    differing = set()
    all_keys = set().union(*(d.keys() for d in dicts))
    for key in all_keys:
        values = [d.get(key) for d in dicts]
        first = values[0]
        if any(value != first for value in values[1:]):
            differing.add(key)
    return differing


def report_duplicate_ids(named_lists, print_ids=False):
    """Print duplicate dok_id values for each named list."""
    any_duplicates = False
    for name, rows in named_lists.items():
        duplicates = duplicate_dok_ids(rows)
        if duplicates:
            any_duplicates = True
            print(f"{name} has {len(duplicates)} duplicate ids")
            if print_ids:
                for dok_id in sorted(duplicates):
                 print(f"  {dok_id}")
    if not any_duplicates:
        print("No duplicate ids found.")


def report_all_duplicates_per_dataset():
    """Iterate canonical datasets and report duplicate dok_id values."""
    for label, dataset in canonical_data.items():
        print(f"{label}:")
        report_duplicate_ids(dataset)
        print()


report_all_duplicates_per_dataset()

# outputs:
# Descriptions:
# No duplicate ids found.
#
# JSON Texts:
# sources/36812-demokratibasen-texts-20250528.jsonl has 1 duplicate ids
#
# CSV Texts:
# No duplicate ids found.
#
# Inferences:
# No duplicate ids found.
#
# Joined Examples:
# training_data/17720-examples-from-prod-20250930.csv has 1837 duplicate ids


# inspect the two cases of duplicate ids found


# Case 1 in JSON Texts:
# sources/36812-demokratibasen-texts-20250528.jsonl has 1 duplicate ids
#     1401a427-3d6e-52b6-a403-490cdb91b0b5
fn = "sources/36812-demokratibasen-texts-20250528.jsonl"
text_data = canonical_json_texts[fn]
dup_id_rows = [row for row in text_data if row.get("dok_id") == "1401a427-3d6e-52b6-a403-490cdb91b0b5"]
for key, val1 in dup_id_rows[0].items():
    val2 = dup_id_rows[1].get(key)
    if val1 != val2:
        print(f"{key}: {val1} != {val2}")
    else:
        print(f"{key} is identical")
# Conclusion: only a single duplicate id - two identical copies of the same document


# Case 2 in Joined Examples:
# training_data/17720-examples-from-prod-20250930.csv has 1837 duplicate ids:

# First define some helper functions

def check_duplicate_ids(example_data):
    duplicate_ids = duplicate_dok_ids(example_data)
    dup_id_rows = {
        dok_id: [row for row in example_data 
                 if row.get("dok_id") == dok_id] 
        for dok_id in duplicate_ids
    }
    return dup_id_rows

    # outputs:
    # num_dups = map(len, dup_id_rows.values())
    # num_dups_counter = Counter(num_dups)
    # print(num_dups_counter)
    # {2: 1314, 3: 201, 4: 122, 6: 74, 5: 69, 13: 50, 7: 4, 9: 1, 15: 1, 8: 1}


def find_differing_keys(dup_rowset):
    keys = dup_rowset[0].keys()
    differing_keys = set()
    for key in keys:
        values = [row.get(key) for row in dup_rowset]
        if len(set(values)) > 1:
            differing_keys.add(key)
    return differing_keys


def report_differing_keys(dup_id_rows):
    for dok_id, dup_rowset in dup_id_rows.items():
        differing_keys = find_differing_keys(dup_rowset)
        if differing_keys:
            print(f"{dok_id} has differing keys: {differing_keys}")


# 1837 duplicate ids - must explore more
fn = "training_data/17720-examples-from-prod-20250930.csv"
joined_example_data = canonical_joined_examples[fn]
dup_id_rows = check_duplicate_ids(joined_example_data)
differing_keys = report_differing_keys(dup_id_rows)
# output:
# 3e05b49f-8e70-5798-aa38-7a123f2676ef has differing keys: {'tekst'}
# all the others have different models ("modell") at most


# list all models ('modell') in joined_example_data
models = set([row.get("modell") for row in joined_example_data])
# output: {'gpt-3.5-turbo', 'gpt-4o-mini'}

# split joined_example_data into two lists: one for each model {'gpt-3.5-turbo', 'gpt-4o-mini'}
gpt_3_5_turbo_data = [row for row in joined_example_data if row.get("modell") == "gpt-3.5-turbo"]
print(f"Number of rows in gpt_3_5_turbo_data: {len(gpt_3_5_turbo_data)}")

gpt_4o_mini_data = [row for row in joined_example_data if row.get("modell") == "gpt-4o-mini"]
print(f"Number of rows in gpt_4o_mini_data: {len(gpt_4o_mini_data)}")

# check them separately for duplicate ids and differing keys
dup_id_rows = check_duplicate_ids(gpt_3_5_turbo_data)
differing_keys = report_differing_keys(dup_id_rows)

dup_id_rows = check_duplicate_ids(gpt_4o_mini_data)
differing_keys = report_differing_keys(dup_id_rows)

# Conclusion: all the differences have disappeared - they were due to different models


# However, for one dok_id, the two models also had different texts

def get_dok_by_id(dok_id, dokumenter):
    for dok in dokumenter:
        if dok.get("dok_id") == dok_id:
            return dok
    raise ValueError(f"No document found with dok_id {dok_id}")

dok_id = "3e05b49f-8e70-5798-aa38-7a123f2676ef"
text_3_5 = get_dok_by_id(dok_id, gpt_3_5_turbo_data)["tekst"]
text_4o = get_dok_by_id(dok_id, gpt_4o_mini_data)["tekst"]

# Conclusion: the texts reflect the same PDF document, 
# but one of the text extractions seems to contain garbage at the end



