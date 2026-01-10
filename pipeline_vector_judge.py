# pipeline_vector_judge.py

import json
import os
import time
import hashlib
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from extraction.constraint_miner import mine_constraints_from_chunk
from utils.llm_client import call_llm
from collections import Counter


# --- CONFIG ---
CHUNKS_FILE = "data/chunks.jsonl"
CAPTION_CLAIMS_FILE = "data/claims_output.json"
OUTPUT_FILE = "data/final_predictions_vector.jsonl"
CACHE_FILE = "data/constraint_cache.json"
VECTOR_STORE = "./data/vector_store.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# --- DIAGNOSTICS ---
def diagnose_retrieval_and_mining(claim_id, hits, mined_results):
    status = Counter()
    for r in mined_results.values():
        if r is None:
            status["api_failure_none"] += 1
        elif isinstance(r, list) and len(r) == 0:
            status["true_empty"] += 1
        else:
            status["non_empty_constraints"] += 1

    print("\n📊 DIAGNOSTICS FOR CLAIM:", claim_id)
    print("   Retrieved chunks:", len(hits))
    print("   Mining outcomes:")
    for k, v in status.items():
        print(f"     - {k}: {v}")
    
    # Print similarity scores to ensure we aren't getting garbage
    sims = [h[0] for h in hits]
    print(f"   Similarity scores: {sims}")


# --- 1. VECTOR ENGINE ---
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
            print("Building vector index (This happens once)...")
            # Load chunks
            with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    self.chunks.append(json.loads(line))
            
            texts = [c['chunk_text'] for c in self.chunks]
            print(f"Embedding {len(texts)} chunks... (Grab a coffee, this takes ~2 mins)")
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            
            with open(VECTOR_STORE, 'wb') as f:
                pickle.dump({'chunks': self.chunks, 'embeddings': self.embeddings}, f)
                
    def search(self, query, top_k=2):
        if not query or not query.strip():
            return []
        query_vec = self.model.encode([query])
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        # Get top indices
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((sims[idx], self.chunks[idx]))
        return results

# --- 2. CACHED MINER ---
def get_constraints_for_chunk(chunk, cache):
    c_id = hashlib.md5(chunk['chunk_text'].encode("utf-8")).hexdigest()
    
    # If in cache and valid, return it
    if c_id in cache:
        return cache[c_id]
    
    # Mine
    print(f"   ⛏️  Mining fresh chunk: {c_id[:8]}...")
    try:
        constraints = mine_constraints_from_chunk(chunk['chunk_text'])
        
        # KEY CHANGE: Only cache if NOT None.
        # If it's [], we cache it (True Empty).
        # If it's valid data, we cache it.
        # If it's None (API Error), we DO NOT cache it.
        if constraints is not None:
            cache[c_id] = constraints
            # Save immediately
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2)
        
        return constraints
    except Exception as e:
        print(f"   ⚠️ Error mining chunk {c_id}: {e}")
        return None

# --- 3. JUDGE ---
def evaluate_consistency(claim, constraints):
    # Filter out None results
    valid_constraints = [c for c in constraints if c is not None]
    
    # Flatten list of lists if necessary
    flat_constraints = []
    for item in valid_constraints:
        if isinstance(item, list):
            flat_constraints.extend(item)
        elif isinstance(item, dict):
            flat_constraints.append(item)

    if not flat_constraints:
        return "consistent", "No constraints found (Default)."
        
    # Format evidence
    evidence_text = ""
    for c in flat_constraints:
        # Handle "Raw Constraints" from our fallback logic
        c_text = c.get('constraint_text', str(c))
        c_source = c.get('source_excerpt', 'Unknown')
        evidence_text += f"- {c_text} (Source: {c_source})\n"
    
    prompt = f"""
    You are a Continuity Editor.
    
    USER CLAIM: "{claim}"
    
    ESTABLISHED FACTS (from the novel):
    {evidence_text}
    
    TASK:
    Does the User Claim CONTRADICT the Established Facts?
    - If the claim explicitly violates a fact, answer "contradict".
    - If the claim fits (or is not mentioned), answer "consistent".
    
    Return JSON: {{ "verdict": "consistent" | "contradict", "reason": "short explanation" }}
    """
    
    try:
        resp = call_llm(prompt)
        if not resp:
             return "consistent", "LLM Judge Failed (Empty Response)"
             
        if "```" in resp: 
            resp = resp.split("```")[1].replace("json", "").strip()
            
        data = json.loads(resp)
        return data.get("verdict", "consistent"), data.get("reason", "")
    except Exception as e:
        print(f"   ⚠️ Judge Error: {e}")
        return "consistent", f"Judge Error: {e}"

def flatten_atomic_claims(claims_dict):
    """
    Turns the structured claims (events, traits) back into a list of strings
    for vector search queries.
    """
    probes = []
    if not claims_dict: return []
    
    # Events
    for e in claims_dict.get('events', []):
        probes.append(e.get('description', ''))
        
    # Traits
    for t in claims_dict.get('traits', []):
        probes.append(f"{t.get('trait', '')} is {t.get('time_scope', '')}")
        
    return [p for p in probes if p]

# --- MAIN ---
def run_vector_pipeline():
    # Setup Engine
    engine = VectorEngine()
    engine.load_or_build_index()
    
    # Load Cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            try:
                cache = json.load(f)
            except:
                cache = {}
    else:
        cache = {}

    # Load Claims
    with open(CAPTION_CLAIMS_FILE, 'r', encoding='utf-8') as f:
        claims_data = json.load(f)
        
    results = []  # <--- INITIALIZED HERE. Safe.

    print(f"Processing {len(claims_data)} user claims with VECTOR SEARCH...")
    
    for i, item in enumerate(claims_data):
        claim_id = item.get('id')
        claim_text = (item.get('input_text') or "").strip()
        
        if not claim_text:
            print(f"⚠️ Empty claim id={claim_id}, skipping.")
            results.append({
                "id": claim_id,
                "input_text": "",
                "verdict": "consistent",
                "reason": "No claim text."
            })
            continue
            
        print(f"\n[{i+1}/{len(claims_data)}] {claim_text[:60]}...")
        
        # 1. Generate Search Probes
        probes = flatten_atomic_claims(item.get('claims', {}))
        if not probes: 
            probes = [claim_text] # Fallback to full text
            
        # 2. Vector Search & Mining
        mined_results = {} # Map chunk_hash -> constraints
        all_constraints = []
        
        # Search for each probe
        for p in probes:
            hits = engine.search(p, top_k=2)
            
            for score, chunk in hits:
                # Deduplicate by content hash
                c_hash = hashlib.md5(chunk['chunk_text'].encode("utf-8")).hexdigest()
                
                if c_hash in mined_results:
                    continue # Already mined this chunk for this claim
                
                constraints = get_constraints_for_chunk(chunk, cache)
                mined_results[c_hash] = constraints
                
                if constraints:
                    all_constraints.append(constraints)

        # Diagnostics
        # Pass a list of hits for diagnostics (just use the last hits from loop)
        diagnose_retrieval_and_mining(claim_id, hits, mined_results)
        
        # 3. Judge
        verdict, reason = evaluate_consistency(claim_text, all_constraints)
        print(f"   ⚖️  {verdict.upper()}")
        
        # 4. Save Result
        results.append({
            "id": claim_id,
            "input_text": claim_text,
            "verdict": verdict,
            "reason": reason
        })
        
        # Periodic Save (Safety)
        if i % 10 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

    # Final Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print("✅ Pipeline Complete.")

if __name__ == "__main__":
    run_vector_pipeline()