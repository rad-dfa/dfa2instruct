#!/usr/bin/env python3
"""
split_dataset.py

Create smaller datasets by randomly sampling examples from an existing pickled
dataset. The script tries to be generic: it supports datasets that are lists,
tuples, numpy arrays, or dicts whose values are same-length sequences/arrays.

Usage examples:
  python split_dataset.py dataset.pkl --sizes 100 500 --out-dir ./out --seed 42

The outputs will be written to the output directory with names derived from
the input file (e.g. dataset_n100.pkl).
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import Any, List


def load_dataset(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_dataset(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def get_length(dataset: Any) -> int:
    try:
        return len(dataset)  # works for list, tuple, numpy arrays, dict (keys)
    except Exception:
        # For dicts where keys are not the items, look for a value with length
        if isinstance(dataset, dict):
            for v in dataset.values():
                try:
                    return len(v)
                except Exception:
                    continue
        raise ValueError("Could not determine dataset length")


def subset_dataset(dataset: Any, indices: List[int]) -> Any:
    dfax_list = list(dataset.keys())
    return {dfax_list[i]: dataset[dfax_list[i]] for i in indices}


def sample_indices(n_total: int, k: int, sampled_indices: set[int], replace: bool, rng: random.Random) -> List[int]:
    if not replace:
        if k > n_total:
            raise ValueError(f"Requested {k} samples but dataset has only {n_total} items (use --replace to allow sampling with replacement)")
        available_indices = list(set(range(n_total)) - sampled_indices)
        if k > len(available_indices):
            raise ValueError(f"Not enough remaining samples to fulfill request of {k} samples without replacement")
        indices = rng.sample(available_indices, k)
        return indices
    else:
        return [rng.randrange(n_total) for _ in range(k)]


def build_out_name(input_path: str, size: int, suffix: str | None = None) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    if suffix:
        return f"{base}{suffix.format(size=size)}"
    return f"{base}_n{size}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a pickled dataset by random sampling")
    parser.add_argument("--dataset", help="Path to input pickled dataset (dataset.pkl)")
    parser.add_argument("--sizes", "-s", nargs="+", type=int, required=True, help="One or more sample sizes to generate")
    parser.add_argument("--out-dir", "-o", default="./", help="Output directory (default:  ./)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default:  42)")
    parser.add_argument("--replace", action="store_true", help="Sample with replacement (allow size > dataset length)")
    parser.add_argument("--suffix", default=None, help="Format suffix for output filenames, use '{size}' in string (default: '_n{size}')")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_dataset(args.dataset)
    n_total = get_length(data)

    assert sum(args.sizes) <= n_total or args.replace, f"Total requested samples ({sum(args.sizes)}) exceeds dataset size ({n_total}) without replacement (use --replace to allow sampling with replacement)"

    rng = random.Random(args.seed)

    sampled_indices = set()

    for size in args.sizes:
        indices = sample_indices(n_total, size, sampled_indices, args.replace, rng)
        sampled_indices.update(indices)
        subset = subset_dataset(data, indices)
        out_basename = build_out_name(args.dataset, size, args.suffix)
        out_path = os.path.join(args.out_dir, out_basename + ".pkl")
        save_dataset(subset, out_path)
        print(f"Wrote {size} samples to {out_path}")


if __name__ == "__main__":
    main()
