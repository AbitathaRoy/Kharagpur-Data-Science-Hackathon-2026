# fix_checkpoint.py
import json
import os

CONSTRAINTS_FILE = "world_constraints.jsonl"
CHECKPOINT_FILE = "processed_chunks.txt"


def main():
    print("--- Fixing Corrupted Checkpoint File ---")

    if not os.path.exists(CONSTRAINTS_FILE):
        print("No constraints file found. Nothing to preserve.")
        return

    # 1. content: Extract all chunk_ids that actually produced data
    proven_ids = set()
    try:
        with open(CONSTRAINTS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "chunk_id" in data:
                            proven_ids.add(data["chunk_id"])
                    except:
                        pass
    except Exception as e:
        print(f"Error reading constraints: {e}")
        return

    print(f"Found {len(proven_ids)} chunks with valid extracted constraints.")

    # 2. Overwrite the checkpoint file
    # This removes all the "silent failures" from the done list
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        for c_id in proven_ids:
            f.write(f"{c_id}\n")

    print(f"✅ fixed! 'processed_chunks.txt' now contains only {len(proven_ids)} IDs.")
    print("You can now run 'pipeline_mining.py'. It will retry the rest.")


if __name__ == "__main__":
    main()