# main.py

import csv
import json
import os
import time
import sys
from dataclasses import asdict, is_dataclass  

# Ensure we can find the modules if running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extraction.caption_claims import extract_caption_claims

# Configuration
INPUT_FILE = "data/train.csv"          
OUTPUT_FILE = "data/claims_output.json"
DELAY_SECONDS = 0.1               

def load_train_data(file_path: str):
    """
    Ingests the specific format of train.csv.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return []

    items = []
    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                text_content = row.get("content", "").strip()
                if not text_content:
                    continue

                items.append({
                    "process_id": row.get("id", str(i)),
                    "text_to_process": text_content,
                    "metadata": {
                        "book_name": row.get("book_name"),
                        "character": row.get("char"),
                        "original_header": row.get("caption"),
                        "label": row.get("label")
                    }
                })
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
        
    return items

def main():
    print("--- Starting Extraction Pipeline ---")
    
    # 1. Ingest Data
    data_items = load_train_data(INPUT_FILE)
    if not data_items:
        print("No data found. Exiting.")
        return

    print(f"Loaded {len(data_items)} rows from {INPUT_FILE}.")
    
    results = []
    
    # 2. Process Loop
    for i, item in enumerate(data_items):
        pid = item["process_id"]
        text = item["text_to_process"]
        meta = item["metadata"]
        
        print(f"[{i+1}/{len(data_items)}] Extracting for ID {pid} ({meta['character']})...")

        try:
            # --- THE CORE CALL ---
            claims_obj = extract_caption_claims(text)
            
            # --- UNIVERSAL SERIALIZER (THE FIX) ---
            if is_dataclass(claims_obj):
                # Case A: Standard Python @dataclass
                claims_dict = asdict(claims_obj)
            else:
                # Case B: Pydantic Model (V1 or V2)
                try:
                    claims_dict = claims_obj.model_dump() # Pydantic V2
                except AttributeError:
                    try:
                        claims_dict = claims_obj.dict()   # Pydantic V1
                    except AttributeError:
                        # Fallback if it's neither (shouldn't happen, but safe)
                        claims_dict = claims_obj.__dict__

            results.append({
                "id": pid,
                "input_text": text,
                "claims": claims_dict,
                "metadata": meta,
                "status": "success"
            })

        except Exception as e:
            print(f"FAILED on ID {pid}: {e}")
            # print(type(claims_obj)) # Uncomment if you need to debug the type
            results.append({
                "id": pid,
                "input_text": text,
                "error": str(e),
                "status": "error"
            })
        
        if DELAY_SECONDS > 0:
            time.sleep(DELAY_SECONDS)

    # 3. Save Output
    print(f"Saving {len(results)} results to {OUTPUT_FILE}...")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()