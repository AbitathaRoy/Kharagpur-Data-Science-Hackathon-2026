# pipeline_vector_judge.py

import json
import os
import time
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.llm_client import call_llm

# --- CONFIG ---
CHUNKS_FILE = "data/chunks.jsonl"
CAPTION_CLAIMS_FILE = "data/claims_output.json"
OUTPUT_FILE = "data/final_predictions_vector.jsonl"
VECTOR_STORE = "./data/vector_store.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieve top 20 chunks (Scoped to the specific book)
# We go deeper because we know they are all from the right book.
TOP_K = 20 

# --- 1. SCOPED VECTOR ENGINE ---
class VectorEngine:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []
        self.embeddings = None

    def load_or_build_index(self):
        if os.path.exists(VECTOR_STORE):
            print("Loading vector index from disk...")
            with open(VECTOR_STORE, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.embeddings = data['embeddings']
        else:
            print("Building vector index...")
            with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    self.chunks.append(json.loads(line))
            
            texts = [c['chunk_text'] for c in self.chunks]
            print(f"Embedding {len(texts)} chunks...")
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            
            with open(VECTOR_STORE, 'wb') as f:
                pickle.dump({'chunks': self.chunks, 'embeddings': self.embeddings}, f)
                
    def search(self, query, book_filter=None, top_k=TOP_K):
        if not query or not query.strip(): return []
        
        # 1. Identify indices for the target book
        if book_filter:
            # Normalize strings for safer matching
            bf = book_filter.lower().strip()
            indices = [
                i for i, c in enumerate(self.chunks) 
                if bf in c.get('metadata', {}).get('book_name', '').lower()
            ]
            
            # Fallback: If filtering removes everything (e.g., name mismatch), search all
            if not indices:
                # print(f"⚠️ Warning: No chunks found for book '{book_filter}'. Searching all.")
                indices = range(len(self.chunks))
        else:
            indices = range(len(self.chunks))
            
        # 2. Encode Query
        query_vec = self.model.encode([query])
        
        # 3. Filter Embeddings & Calculate Similarity
        # We slice the master embeddings array to only include target book chunks
        filtered_embeddings = self.embeddings[indices]
        
        if len(filtered_embeddings) == 0:
            return []

        sims = cosine_similarity(query_vec, filtered_embeddings)[0]
        
        # 4. Get Top K (Relative to the filtered list)
        # Ensure we don't ask for more chunks than exist in the book
        k = min(top_k, len(sims))
        top_relative_indices = np.argsort(sims)[-k:][::-1]
        
        results = []
        for rel_idx in top_relative_indices:
            original_idx = indices[rel_idx] # Map back to global ID
            results.append((sims[rel_idx], self.chunks[original_idx]))
            
        return results

# --- 2. THE DETECTIVE JUDGE ---
def evaluate_claim_raw(claim, chunks):
    context_text = ""
    for i, (score, chunk) in enumerate(chunks):
        text = chunk.get('chunk_text', '').replace("\n", " ")
        context_text += f"[{i+1}] {text}\n\n"

    # We use a "Detective" persona to encourage finding the specific details
    prompt = f"""
    You are a Detective verifying a witness statement against the official case files.
    
    WITNESS CLAIM: "{claim}"
    
    OFFICIAL CASE FILES (NOVEL TEXT):
    {context_text}
    
    INSTRUCTIONS:
    1. Search the Case Files for the specific events/characters mentioned in the Claim.
    2. **Compare Details:** Look closely at names, dates, causes of death, and relationships.
    3. **Verdict Logic:**
       - **CONTRADICT**: If the text explicitly tells a *different* story (e.g., Claim says "Shot", Text says "Stabbed").
       - **CONTRADICT**: If the Claim says X happened, but the Text says Y happened *instead*.
       - **CONSISTENT**: If the Claim is supported by the text.
       - **CONSISTENT**: If the Claim adds extra details that do *not* conflict with the text (Silence is not a lie).
    
    Return JSON: {{ "verdict": "consistent" | "contradict", "reason": "Citing specific passage [x]" }}
    """
    
    try:
        resp = call_llm(prompt) 
        if not resp: return "consistent", "LLM Error"
        if "```" in resp: resp = resp.split("```")[1].replace("json", "").strip()
        data = json.loads(resp)
        return data.get("verdict", "consistent"), data.get("reason", "")
    except Exception as e:
        return "consistent", f"Judge Error: {e}"

def flatten_atomic_claims(claims_dict):
    if not claims_dict: return ""
    text = ""
    for e in claims_dict.get('events', []): text += e.get('description', '') + " "
    return text.strip()

# --- MAIN WITH RESUME CAPABILITY ---
def run_pipeline():
    engine = VectorEngine()
    engine.load_or_build_index()
    
    with open(CAPTION_CLAIMS_FILE, 'r', encoding='utf-8') as f: 
        claims_data = json.load(f)
        
    # --- RESUME LOGIC ---
    processed_ids = set()
    results = []
    
    # Check if output file exists and load existing results
    if os.path.exists(OUTPUT_FILE):
        print(f"🔄 Found existing output file: {OUTPUT_FILE}")
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                results = existing_data
                processed_ids = {item['id'] for item in existing_data}
            print(f"   Skipping {len(processed_ids)} already processed claims.")
        except json.JSONDecodeError:
            print("   ⚠️ Output file corrupted or empty. Starting fresh.")

    print(f"Processing {len(claims_data)} claims with SCOPED RAG (Top-{TOP_K})...")
    
    for i, item in enumerate(claims_data):
        claim_id = item.get('id')
        
        # SKIP if already done
        if claim_id in processed_ids:
            continue

        claim_text = (item.get('input_text') or "").strip()
        book_name = item.get('metadata', {}).get('book_name')
        
        if not claim_text: continue
        
        print(f"\n[{i+1}/{len(claims_data)}] ID {claim_id} ({book_name})...")
        
        # 1. Retrieval
        search_query = claim_text + " " + flatten_atomic_claims(item.get('claims', {}))
        hits = engine.search(search_query, book_filter=book_name, top_k=TOP_K)
        
        # 2. Judgment
        verdict, reason = evaluate_claim_raw(claim_text, hits)
        
        print(f"   🔍 Retrieved {len(hits)} chunks (Scoped).")
        print(f"   ⚖️  Verdict: {verdict.upper()}")
        
        results.append({
            "id": claim_id,
            "input_text": claim_text,
            "verdict": verdict,
            "reason": reason
        })
        
        # Save every step so we don't lose progress on crash
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f: 
            json.dump(results, f, indent=2)

    print("✅ Pipeline Complete.")

if __name__ == "__main__":
    run_pipeline()