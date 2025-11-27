"""Analyses duplicate dok_id values in the canonical datasets."""


import itertools

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


def remove_duplicate_objects(dataset):
    """Return dataset with duplicate dict entries removed (preserving order)."""
    unique_rows = []
    seen = set()

    for row in dataset:
        # Row signature: tuple of key/value pairs sorted by key to ensure determinism.
        signature = tuple(sorted(row.items()))
        if signature in seen:
            continue
        seen.add(signature)
        unique_rows.append(row)

    return unique_rows


def create_datasets_per_dataset_type():
    canonical_datasets = {}
    for label, dataset in canonical_data.items():
        concat_dataset = list(itertools.chain(*dataset.values()))
        canonical_datasets[label] = remove_duplicate_objects(concat_dataset)
    return canonical_datasets


def report_all_duplicates_per_dataset(canonical_datasets):
    """Iterate canonical datasets and report duplicate dok_id values."""
    for label, joined_dataset in canonical_datasets.items():
        print(f"{label}:")
        report_duplicate_ids({f"Merged {label}": joined_dataset})
        print()


canonical_datasets = create_datasets_per_dataset_type()
report_all_duplicates_per_dataset(canonical_datasets)


# output:
# Descriptions:
# Merged Descriptions has 664 duplicate ids
# NEED TO CHECK
# 
# Inferences:
# Merged Inferences has 29132 duplicate ids
# NEED TO CHECK
# 
# Joined Examples:
# Merged Joined Examples has 240 duplicate ids
# NEED TO CHECK



# Case 1
# Descriptions:
# Merged Descriptions has 664 duplicate ids

dataset = canonical_datasets["Descriptions"]



# Case 2
# Inferences:
# Merged Inferences has 29132 duplicate ids

inference_datasets = canonical_data["Inferences"]

labelled_dataset_pairs = itertools.combinations(inference_datasets.items(), 2)
paired_datasets = {}
for lab_ds_pair1, lab_ds_pair2 in labelled_dataset_pairs:
    label1, dataset1 = lab_ds_pair1
    label2, dataset2 = lab_ds_pair2
    paired_datasets[f'{label1} vs {label2}'] = list(itertools.chain(dataset1, dataset2))

report_all_duplicates_per_dataset(paired_datasets)



# Case 3
# Joined Examples:
# Merged Joined Examples has 240 duplicate ids

dataset = canonical_datasets["Joined Examples"]

models = set([row.get("modell") for row in dataset])
# output: {'gpt-3.5-turbo', 'gpt-4o-mini'}

# split joined_example_data into two lists: one for each model {'gpt-3.5-turbo', 'gpt-4o-mini'}
gpt_3_5_turbo_data = [row for row in dataset if row.get("modell") == "gpt-3.5-turbo"]
print(f"Number of rows in gpt_3_5_turbo_data: {len(gpt_3_5_turbo_data)}")

gpt_4o_mini_data = [row for row in dataset if row.get("modell") == "gpt-4o-mini"]
print(f"Number of rows in gpt_4o_mini_data: {len(gpt_4o_mini_data)}")

# Number of rows in gpt_3_5_turbo_data: 110
# Number of rows in gpt_4o_mini_data: 31204