# pipeline_knowledge.py

import pathway as pw
import os
from extraction.constraint_miner import mine_constraints_from_chunk

# --- CONFIGURATION ---
DATA_DIR = "./data/novels/"
OUTPUT_FILE = "./data/world_constraints.jsonl"

# --- HELPER: CHUNKING UDF ---
def split_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """
    Splits text into overlapping chunks.
    Pathway will 'explode' (flatten) the list this returns into separate rows.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        # Move forward by (chunk_size - overlap)
        start += (chunk_size - overlap)
        
    return chunks

def run_pipeline():
    # 1. Ingest: Monitor the directory for .txt files
    # mode="static" reads once. mode="streaming" watches for updates.
    # For phase 3A, "static" is safer to ensure it finishes.
    files = pw.io.fs.read(
        DATA_DIR, 
        format="plaintext", 
        mode="static",
        with_metadata=True
    )

    # 2. Chunking: Split the 'data' (full text) into chunks
    # We use 'flat_map' to turn 1 row (novel) into N rows (chunks)
    chunks = files.select(
        book_name=pw.this.path, # Metadata: filename
        chunk_text=pw.apply(split_into_chunks, pw.this.data)
    ).flatten(pw.this.chunk_text)

    # 3. Mining: Apply the LLM UDF to each chunk
    # This also produces a list of constraints per chunk, so we flatten again.
    extracted_raw = chunks.select(
        book_name=pw.this.book_name,
        constraint_list=pw.apply(mine_constraints_from_chunk, pw.this.chunk_text)
    ).flatten(pw.this.constraint_list)

    # 4. Unpack: Turn the JSON dicts into proper Pathway columns
    constraints = extracted_raw.select(
        book_name=pw.this.book_name,
        character=pw.this.constraint_list["character"],
        constraint_type=pw.this.constraint_list["constraint_type"],
        constraint_text=pw.this.constraint_list["constraint_text"],
        time_scope=pw.this.constraint_list["time_scope"],
        source_excerpt=pw.this.constraint_list["source_excerpt"]
    )

    # 5. Deduplicate (Step 5 of Plan)
    # If the EXACT same constraint text appears for the same character/type, keep one.
    deduplicated = constraints.groupby(
        pw.this.book_name, 
        pw.this.character, 
        pw.this.constraint_type, 
        pw.this.constraint_text
    ).reduce(
        book_name=pw.this.book_name,
        character=pw.this.character,
        constraint_type=pw.this.constraint_type,
        constraint_text=pw.this.constraint_text,
        time_scope=pw.reducers.first(pw.this.time_scope),    # Keep the first time_scope found
        source_excerpt=pw.reducers.first(pw.this.source_excerpt) # Keep the first proof found
    )

    # 6. Output: Write to JSONL
    pw.io.jsonlines.write(deduplicated, OUTPUT_FILE)
    
    print(f"Pipeline defined. Monitoring {DATA_DIR}...")
    
    # Run the pipeline (since we used mode="static", it will process and exit)
    pw.run()

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    run_pipeline()