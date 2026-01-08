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
from collections import Counter, defaultdict


# --- CONFIG ---
CHUNKS_FILE = "data/chunks.jsonl"
CLAIMS_FILE = "./data/train.csv"  # Using train.csv directly to iterate IDs
OUTPUT_FILE = "data/final_predictions_vector.jsonl"
CACHE_FILE = "data/constraint_cache.json"
VECTOR_STORE = "./data/vector_store.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight, effective

def diagnose_retrieval_and_mining(claim_id, hits, mined_results):
    unique_chunk_ids = list(mined_results.keys())

    status_counter = Counter()
    for cid, result in mined_results.items():
        if result is None:
            status_counter["api_failure_none"] += 1
        elif isinstance(result, list) and len(result) == 0:
            status_counter["true_empty"] += 1
        else:
            status_counter["non_empty_constraints"] += 1

    print("\n📊 DIAGNOSTICS FOR CLAIM:", claim_id)
    print("   Retrieved chunks (raw):", len(hits))
    print("   Retrieved chunks (unique):", len(unique_chunk_ids))
    print("   Chunk IDs:", [cid[:8] for cid in unique_chunk_ids])

    print("   Mining outcomes:")
    for k, v in status_counter.items():
        print(f"     - {k}: {v}")

    print("   Similarity scores:",
          [round(h[0], 4) for h in hits])


# --- 1. VECTOR ENGINE ---
class VectorEngine:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []
        self.embeddings = None

    def load_or_build_index(self):
        # Check if we have a saved index
        if os.path.exists(VECTOR_STORE):
            print("Loading vector index from disk...")
            with open(VECTOR_STORE, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.embeddings = data['embeddings']
        else:
            print("Building vector index (This happens once)...")
            # Load chunks
            with open(CHUNKS_FILE, 'r') as f:
                for line in f:
                    self.chunks.append(json.loads(line))

            texts = [c['chunk_text'] for c in self.chunks]
            print(f"Embedding {len(texts)} chunks... (Grab a coffee, this takes ~2 mins)")
            self.embeddings = self.model.encode(texts, show_progress_bar=True)

            # Save
            with open(VECTOR_STORE, 'wb') as f:
                pickle.dump({'chunks': self.chunks, 'embeddings': self.embeddings}, f)

    def search(self, query, top_k=3):
        # Embed the query
        query_vec = self.model.encode([query])
        # Calculate similarity
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        # Get top K indices
        top_indices = np.argsort(sims)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((sims[idx], self.chunks[idx]))
        return results


# --- 2. CACHED MINER (Reused) ---
def get_constraints_for_chunk(chunk, cache):
    c_id = hashlib.md5(chunk['chunk_text'].encode()).hexdigest()

    # Cached true-empty OR non-empty
    if c_id in cache:
        return cache[c_id]

    print(f"   ⛏️  Mining fresh chunk: {c_id[:8]}...")
    try:
        constraints = mine_constraints_from_chunk(chunk['chunk_text'])

        if constraints is None:
            # API failure → do NOT cache
            return None

        # True empty OR non-empty → cache both
        cache[c_id] = constraints
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)

        return constraints

    except Exception as e:
        print("   ❌ Miner exception:", e)
        return None



# --- 3. JUDGE (Reused) ---
def evaluate_consistency(claim, constraints):
    if not constraints:
        return "consistent", "No explicit constraints found."

    # --- Evidence analysis ---
    hard_constraints = [
        c for c in constraints
        if c.get("constraint_type") in ("hard_fact", "temporal_lock")
    ]

    sufficient_evidence = (
        len(hard_constraints) >= 1 or
        len(constraints) >= 2
    )

    evidence_text = "\n".join(
        f"- {c.get('constraint_text', '')}"
        for c in constraints
    )

    prompt = f"""
You are a strict continuity editor.

USER CLAIM:
"{claim}"

ESTABLISHED FACTS:
{evidence_text}

Decide carefully:

- Return "contradict" ONLY if the claim clearly violates the facts.
- If evidence is weak, incomplete, or indirect, return "consistent".

Return STRICT JSON:
{{ "verdict": "consistent" | "contradict", "reason": "short explanation" }}
"""

    resp = call_llm(prompt)

    try:
        if "```" in resp:
            resp = resp.split("```")[1].replace("json", "")

        data = json.loads(resp)
        verdict = data.get("verdict", "consistent")

        # --- 🔒 CONTRADICTION GATE ---
        if verdict == "contradict" and not sufficient_evidence:
            return "consistent", "Insufficient evidence to confirm contradiction."

        if verdict not in ("consistent", "contradict"):
            verdict = "consistent"

        return verdict, data.get("reason", "")

    except Exception:
        return "consistent", "Judge parsing error."



# --- MAIN ---
def run_vector_pipeline():
    # Setup Engine
    engine = VectorEngine()
    engine.load_or_build_index()

    # Load Claims (CSV)
    import pandas as pd
    df = pd.read_csv(CLAIMS_FILE)
    claims_data = df.to_dict('records')

    # Load Cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    results = []
    print(f"Processing {len(claims_data)} user claims with VECTOR SEARCH...")

    for i, item in enumerate(claims_data):
        input_text = str(item.get('content', '')).strip()
        if not input_text:
            raise ValueError(
                f"Empty claim text for row id={item.get('id')}. "
                "Check CSV column mapping."
            )

        print(f"\n[{i + 1}/{len(claims_data)}] Claim: {input_text[:50]}...")

        # 1. VECTOR SEARCH
        # More evidence needed for contradiction detection
        top_k = 7 if len(input_text) > 80 else 5
        hits = engine.search(input_text, top_k=top_k)

        print("   🧪 Query preview:", input_text[:80])
        print("   🧪 Top chunk previews:")
        for score, ch in hits:
            print(f"      ({score:.3f}) {ch['chunk_text'][:80].replace(chr(10), ' ')}")

        relevant_chunks = [h[1] for h in hits]
        print(f"   🔍 Found 3 chunks (Sim: {hits[0][0]:.4f})")

        # 2. MINE (WITH DIAGNOSTICS)
        all_constraints = []
        mined_results = {}

        for ch in relevant_chunks:
            c_id = hashlib.md5(ch['chunk_text'].encode()).hexdigest()

            if c_id in mined_results:
                continue  # safety against duplicates

            result = get_constraints_for_chunk(ch, cache)
            mined_results[c_id] = result

            if result:
                all_constraints.extend(result)

        # 3. JUDGE
        diagnose_retrieval_and_mining(item['id'], hits, mined_results)
        verdict, reason = evaluate_consistency(input_text, all_constraints)
        print(f"   ⚖️  Verdict: {verdict.upper()}")

        results.append({
            "id": item['id'],
            "input_text": input_text,
            "verdict": verdict,
            "reason": reason
        })
        time.sleep(1)  # Safety sleep

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_vector_pipeline()