# pipeline_chunking.py
import pathway as pw
import os
import re

DATA_DIR = "./data/novels/"
OUTPUT_FILE = "data/chunks.jsonl"


def clean_and_split_sentences(text: str) -> list[str]:
    if not text: return []

    # 1. Decode bytes if necessary (Crucial for format="binary")
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError:
            return []  # Skip binary garbage if any

    # 2. Semantic Cleaning (same as before)
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]


def semantic_chunking(text_data, target_size: int = 1200, overlap_sentences: int = 2) -> list[str]:
    # Pass the raw data (bytes or str) to the splitter
    sentences = clean_and_split_sentences(text_data)
    if not sentences: return []

    chunks = []
    current_chunk = []
    current_len = 0
    overlap_buffer = []

    for sentence in sentences:
        s_len = len(sentence)

        if current_len + s_len > target_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_buffer = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = list(overlap_buffer)
            current_len = sum(len(s) for s in current_chunk) + len(current_chunk)

        current_chunk.append(sentence)
        current_len += s_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def run_chunking():
    print("--- Starting Phase 3A.1: Semantic Chunking (Whole File Mode) ---")

    # 1. Setup paths
    input_glob = os.path.join(DATA_DIR, "*.txt")  # Fixes the DeprecationWarning

    if os.path.abspath(OUTPUT_FILE).startswith(os.path.abspath(DATA_DIR)):
        print(f"❌ Error: Output {OUTPUT_FILE} is inside input folder.")
        return

    # 2. Ingest as BINARY to get the whole file content in one row
    files = pw.io.fs.read(
        input_glob,  # Use glob pattern directly
        format="binary",  # <--- THE FIX: Reads whole file as bytes
        mode="static",
        with_metadata=True
    )

    # 3. Debug: Expect small number of rows now (e.g., 2 novels = 2 rows)
    debug_files = files.select(path=pw.this._metadata["path"])
    pw.io.csv.write(debug_files, "data/debug_files_list.csv")

    # 4. Split
    # Since 'data' is now the whole book, semantic_chunking runs once per book.
    chunks = files.select(
        book_name=pw.this._metadata["path"],
        chunk_text=pw.apply(semantic_chunking, pw.this.data)
    ).flatten(pw.this.chunk_text)

    # 5. Save
    pw.io.jsonlines.write(chunks, OUTPUT_FILE)

    print(f"Reading from {input_glob}...")
    pw.run()
    print(f"Chunks saved to {OUTPUT_FILE}")
    print("Check 'data/debug_files_list.csv'. It should now have very few rows (one per book).")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    run_chunking()