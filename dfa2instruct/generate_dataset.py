import argparse
import dill
from pathlib import Path
import ollama


def generate_dataset(storage_dir: str = "storage", output_file: str = "dataset.pkl", llm: str = "qwen3-embedding:4b", dimensions: int = 32):
    """
    Combine all pickle files in the storage directory into a single dataset.
    
    Args:
        storage_dir: Directory containing pickle files to combine (default: "storage")
        output_file: Name of the output pickle file (default: "dataset.pkl")
        dimensions: Embedding dimensions for Qwen3-Embedding 4b (default: 32)
        llm: LLM model name for embeddings (default: "qwen3-embedding:4b")
    """
    storage_path = Path(storage_dir)
    
    if not storage_path.exists():
        print(f"Error: Storage directory '{storage_dir}' does not exist.")
        return
    
    # Find all pickle files
    pkl_files = sorted(storage_path.glob("*.pkl"))
    
    if not pkl_files:
        print(f"No pickle files found in '{storage_dir}'")
        return
    
    print(f"Found {len(pkl_files)} pickle files to combine...")
    
    dataset = {}
    
    for pkl_file in pkl_files:
        print(f"Loading {pkl_file.name}...", end="\r", flush=True)
        try:
            with open(pkl_file, "rb") as f:
                data = dill.load(f)
                dataset.update(data)
        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")
            continue
    
    print(f"Loaded all files. Total items: {len(dataset)}".ljust(60))

    # Pass all values of dataset through Qwen3-Embedding 4b
    print("Generating embeddings with Qwen3-Embedding 4b...")
    
    embedded_dataset = {}
    for i, (dfax, instruct) in enumerate(dataset.items()):
        print(f"Embedding {i + 1}/{len(dataset)}", end="\r", flush=True)
        try:

            response = ollama.embed(llm, instruct, dimensions=dimensions)
            embedding = response["embeddings"][0]
            
            # Store original value with embedding
            embedded_dataset[dfax] = (instruct, embedding)

        except Exception as e:
            print(f"Error embedding item {dfax}: {e}")
            # Keep original data if embedding fails
            embedded_dataset[dfax] = instruct
            continue
    
    print(f"Embedding complete.".ljust(60))
    # Save dataset
    with open(output_file, "wb") as f:
        dill.dump(embedded_dataset, f)
    
    print(f"Dataset saved to '{output_file}'")
    print(f"Total samples in dataset: {len(embedded_dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine all pickle files in a given storage directory")
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="storage",
        help="Directory containing pickle files (default: storage)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.pkl",
        help="Output pickle file name (default: dataset.pkl)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        nargs="?",
        default="qwen3-embedding:4b",
        help="LLM model name (default: qwen3-embedding:4b)"
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=32,
        help="Embedding dimensions for Qwen3-Embedding 4b (default: 32)"
    )
    args = parser.parse_args()
    
    generate_dataset(storage_dir=args.storage_dir, output_file=args.output, llm=args.llm, dimensions=args.dimensions)

