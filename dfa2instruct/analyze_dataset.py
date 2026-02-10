import argparse
import dill
import json
import numpy as np
from pathlib import Path
import random


def load_dataset(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(p, "rb") as f:
        data = dill.load(f)
    return data


def analyze_dataset(dataset, rng_seed=0):
    total = len(dataset)

    embeddings = []
    keys_with_emb = []
    instr_lengths = []
    missing = 0

    for k, v in dataset.items():
        # expected stored as (instruct, embedding) by generate_dataset
        if isinstance(v, tuple) and len(v) == 2:
            instruct, emb = v
            try:
                arr = np.asarray(emb, dtype=float)
                if arr.ndim == 1:
                    embeddings.append(arr)
                    keys_with_emb.append(k)
                else:
                    # unexpected shape
                    missing += 1
            except Exception:
                missing += 1
            instr_lengths.append(len(instruct) if isinstance(instruct, str) else None)
        else:
            missing += 1
            # if v is just the instruction string, record its length
            if isinstance(v, str):
                instr_lengths.append(len(v))
            else:
                instr_lengths.append(None)

    n_with = len(embeddings)
    n_missing = total - n_with

    report = {
        "total_items": int(total),
        "items_with_embedding": int(n_with),
        "items_missing_embedding": int(n_missing),
    }

    n_states = [dfax.n_states for dfax in dataset]
    report["n_states_max"] = np.max(n_states)
    report["n_states_min"] = np.min(n_states)
    report["n_states_mean"] = np.mean(n_states)
    report["n_states_std"] = np.std(n_states)

    if n_with == 0:
        return report

    E = np.vstack(embeddings)
    dims = E.shape[1]
    norms = np.linalg.norm(E, axis=1)

    def stats(a):
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "median": float(np.median(a)),
        }

    report["embedding_dim"] = int(dims)
    report["norms"] = stats(norms)

    # instruction length stats (where available)
    instr_lens = [l for l in instr_lengths if isinstance(l, int)]
    if instr_lens:
        report["instruction_length"] = stats(np.array(instr_lens, dtype=float))

    # Cosine similarity: either full matrix if small, or sampled pairs
    normalized = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    n = n_with
    max_full = 2000
    if n <= max_full:
        sim = normalized @ normalized.T
        # extract upper triangle without diagonal
        iu = np.triu_indices(n, k=1)
        vals = sim[iu]
        report["cosine_similarity"] = stats(vals)

    else:
        # sample random pairs
        rng = random.Random(rng_seed)
        m = min(total, n * (n - 1) // 2)
        sims = []
        for _ in range(m):
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
            sims.append(float(np.dot(normalized[i], normalized[j])))
        report["cosine_similarity_sampled"] = stats(np.array(sims, dtype=float))

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze embedded DFAx dataset")
    parser.add_argument("--input", "-i", type=str, default="dataset.pkl", help="Input dill file (default: dataset.pkl)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional JSON output file for the report")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    args = parser.parse_args()

    data = load_dataset(args.input)
    report = analyze_dataset(data, rng_seed=args.seed)

    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
