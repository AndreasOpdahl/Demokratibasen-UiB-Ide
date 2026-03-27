"""
Utility functions for managing fixed NLI faithfulness evaluation subset.

Creates or loads a fixed subset of example indices so NLI scores are comparable
across checkpoint runs. Default subset size is 100. Use ``subset_size >= total_examples``
(typically ``subset_size == val_data_size``) to run NLI on the full eval set.
Smaller subsets use a seeded random draw (seed 42) or the first N rows when
extending eval size (use_first_n_for_extended).
"""

import json
import os
import random
from typing import List, Optional, Tuple

NLI_FIXED_SUBSET_SEED = 42  # Fixed seed for reproducible random subsets
NLI_DEFAULT_SUBSET_SIZE = 100
# Legacy name; kept for imports expecting the old constant (was 500).
NLI_FIXED_SUBSET_SIZE = 500


def get_nli_subset_file_path(model_dir: str) -> str:
    """Path to `all_eval_results/nli_fixed_subset_indices.json` under model_dir."""
    return os.path.join(model_dir, "all_eval_results", "nli_fixed_subset_indices.json")


def _load_nli_subset_json(model_dir: str) -> Optional[dict]:
    subset_file = get_nli_subset_file_path(model_dir)
    if not os.path.exists(subset_file):
        return None
    try:
        with open(subset_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _compute_nli_indices(
    total_examples: int,
    nli_subset_size: Optional[int],
    seed: int,
    use_first_n_for_extended: bool,
) -> List[int]:
    """Deterministic indices for NLI with monotonic nesting guarantee.

    Uses a seeded permutation of [0..total_examples): the first *k* elements of
    the permutation form the subset of size *k*.  Because the permutation is
    fixed, ``indices(k1) ⊂ indices(k2)`` for any ``k1 < k2``, which allows
    incremental faithfulness evaluation — results computed for a smaller subset
    are always reusable when expanding to a larger one.
    """
    if nli_subset_size is None:
        nli_subset_size = NLI_DEFAULT_SUBSET_SIZE
    if use_first_n_for_extended and total_examples > nli_subset_size:
        return list(range(min(nli_subset_size, total_examples)))
    if nli_subset_size >= total_examples:
        return list(range(total_examples))
    pool = list(range(total_examples))
    rng = random.Random(seed)
    rng.shuffle(pool)
    return sorted(pool[:nli_subset_size])


def _stored_subset_matches(
    data: dict,
    total_examples: int,
    nli_subset_size: int,
    use_first_n_for_extended: bool,
    seed: int,
) -> bool:
    if data.get("total_examples") != total_examples:
        return False
    indices = data.get("indices")
    if not indices or not isinstance(indices, list):
        return False
    if not all(isinstance(i, int) and 0 <= i < total_examples for i in indices):
        return False

    if "nli_subset_size" in data:
        stored = data.get("nli_subset_size")
        if stored is not None:
            if stored != nli_subset_size:
                return False
            if data.get("use_first_n_for_extended", False) != use_first_n_for_extended:
                return False
            if data.get("seed", seed) != seed:
                return False
            # Metadata matches — verify actual indices against current algorithm
            # (guards against algorithm changes, e.g. sample → shuffle-permutation)
            expected = _compute_nli_indices(
                total_examples, nli_subset_size, seed, use_first_n_for_extended
            )
            return indices == expected
        # Key present, value null: legacy "full val" save
        if len(indices) == total_examples:
            return (
                nli_subset_size >= total_examples
                and indices == list(range(total_examples))
            )

    expected = _compute_nli_indices(
        total_examples, nli_subset_size, seed, use_first_n_for_extended
    )
    return indices == expected


def _save_nli_subset_file(
    model_dir: str,
    indices: List[int],
    total_examples: int,
    nli_subset_size: int,
    use_first_n_for_extended: bool,
    seed: int,
) -> None:
    subset_file = get_nli_subset_file_path(model_dir)
    os.makedirs(os.path.dirname(subset_file), exist_ok=True)
    with open(subset_file, "w") as f:
        json.dump(
            {
                "nli_subset_size": nli_subset_size,
                "use_first_n_for_extended": use_first_n_for_extended,
                "total_examples": total_examples,
                "seed": seed,
                "subset_size": len(indices),
                "indices": indices,
            },
            f,
            indent=2,
        )


def create_fixed_nli_subset(
    total_examples: int,
    subset_size: Optional[int] = None,
    seed: int = NLI_FIXED_SUBSET_SEED,
    model_dir: Optional[str] = None,
) -> List[int]:
    """Create indices (subset_size None → default 100). Full set when subset_size >= total."""
    eff = NLI_DEFAULT_SUBSET_SIZE if subset_size is None else subset_size
    indices = _compute_nli_indices(
        total_examples, eff, seed, use_first_n_for_extended=False
    )
    if model_dir:
        _save_nli_subset_file(
            model_dir,
            indices,
            total_examples,
            eff,
            False,
            seed,
        )
    return indices


def load_fixed_nli_subset(model_dir: str) -> Optional[List[int]]:
    data = _load_nli_subset_json(model_dir)
    if data is None:
        return None
    return data.get("indices")


def get_or_create_fixed_nli_subset(
    total_examples: int,
    model_dir: str,
    subset_size: Optional[int] = None,
    seed: int = NLI_FIXED_SUBSET_SEED,
    use_first_n_for_extended: bool = False,
) -> List[int]:
    """Load existing NLI indices or create and save them.

    subset_size:
        None — use ``NLI_DEFAULT_SUBSET_SIZE`` (100).
        int — at most this many rows; use ``subset_size >= total_examples`` (typically
        equal to ``val_data_size``) to include the full eval set.
    """
    eff = NLI_DEFAULT_SUBSET_SIZE if subset_size is None else subset_size
    data = _load_nli_subset_json(model_dir)
    if data and _stored_subset_matches(
        data, total_examples, eff, use_first_n_for_extended, seed
    ):
        return data["indices"]

    indices = _compute_nli_indices(
        total_examples, eff, seed, use_first_n_for_extended
    )
    _save_nli_subset_file(
        model_dir, indices, total_examples, eff, use_first_n_for_extended, seed
    )
    return indices


def apply_fixed_subset(
    input_texts: List[str],
    prediction_texts: List[str],
    reference_texts: List[str],
    indices: List[int],
) -> Tuple[List[str], List[str], List[str]]:
    """Filter parallel lists to the given indices (bounds-checked)."""
    nli_input_texts = [input_texts[i] for i in indices if i < len(input_texts)]
    nli_prediction_texts = [prediction_texts[i] for i in indices if i < len(prediction_texts)]
    nli_reference_texts = [reference_texts[i] for i in indices if i < len(reference_texts)]
    return nli_input_texts, nli_prediction_texts, nli_reference_texts
