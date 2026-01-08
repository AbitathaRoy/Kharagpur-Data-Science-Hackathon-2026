# pipeline_mining.py

import json
import os
import time
import hashlib
from extraction.constraint_miner import mine_constraints_from_chunk

INPUT_FILE = "chunks.jsonl"
OUTPUT_FILE = "world_constraints.jsonl"
CHECKPOINT_FILE = "processed_chunks.txt"

# --- RATE LIMIT CONFIG ---
SLEEP_SECONDS = 2.0  # Slow down to 30 RPM (Groq limit is usually 30 RPM/14k TPM)


def get_chunk_id(book_name, text):
    """Creates a unique ID for the chunk to track progress."""
    content = f"{book_name}{text[:20]}"
    return hashlib.md5(content.encode()).hexdigest()


def load_processed_ids():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, 'r') as f:
        return set(line.strip() for line in f)


def save_processed_id(chunk_id):
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(f"{chunk_id}\n")


def main():
    print("--- Starting Phase 3A.2: Mining ---")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run pipeline_chunking.py first.")
        return

    # 1. Load Checkpoints
    processed_ids = load_processed_ids()
    print(f"Found {len(processed_ids)} chunks already processed.")

    # 2. Read Chunks
    chunks_to_process = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip():
                chunks_to_process.append(json.loads(line))

    print(f"Total chunks in file: {len(chunks_to_process)}")

    # 3. Process Loop
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
        for i, chunk in enumerate(chunks_to_process):
            book = chunk.get("book_name")
            text = chunk.get("chunk_text")
            c_id = get_chunk_id(book, text)

            # SKIP if done
            if c_id in processed_ids:
                continue

            print(f"[{i + 1}/{len(chunks_to_process)}] Mining chunk from {book}...")

            try:
                # CALL LLM
                constraints = mine_constraints_from_chunk(text)

                # --- THE FIX ---
                if constraints is None:
                    print(f"⚠️ Skipping save for Chunk {c_id} due to API failure.")
                    # Sleep longer (backoff) because we probably hit a rate limit
                    time.sleep(10)
                    continue  # <--- CONTINUE WITHOUT SAVING ID
                # ----------------

                # Write results immediately
                for c in constraints:
                    c["book_name"] = book
                    c["chunk_id"] = c_id
                    json.dump(c, out_f)
                    out_f.write("\n")

                # ONLY save ID if successful
                save_processed_id(c_id)

                # Normal rate limit sleep
                time.sleep(SLEEP_SECONDS)

            except Exception as e:
                print(f"❌ Failed on chunk {c_id}: {e}")
                time.sleep(5)

    print("--- Mining Complete ---")


if __name__ == "__main__":
    main()