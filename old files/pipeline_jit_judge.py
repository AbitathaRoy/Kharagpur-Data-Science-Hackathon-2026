# pipeline_jit_judge.py
import json
import os
import re
import time
import hashlib
from extraction.constraint_miner import mine_constraints_from_chunk
# You will need a judge function (we'll draft a simple one inside)
from utils.llm_client import call_llm

# --- CONFIG ---
CHUNKS_FILE = "chunks.jsonl"
CLAIMS_FILE = "claims_output.json"  # Input (from Phase 2)
OUTPUT_FILE = "final_predictions.jsonl"
CACHE_FILE = "constraint_cache.json"


# --- 1. LOAD THE BOOK (Raw Chunks) ---
def load_chunks():
    chunks = []
    print("Loading chunks...")
    with open(CHUNKS_FILE, 'r') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


# --- 2. RETRIEVAL ENGINE (BM25-ish) ---
def find_relevant_chunks(claim_text, chunks, top_k=3):
    """
    Simple keyword overlap search.
    """
    claim_words = set(re.findall(r'\w+', claim_text.lower()))
    # Remove stopwords
    stopwords = {"the", "and", "is", "in", "to", "of", "a", "he", "she", "was"}
    claim_words = claim_words - stopwords

    scored_chunks = []
    for chunk in chunks:
        text = chunk['chunk_text'].lower()
        score = 0
        for word in claim_words:
            if word in text:
                score += 1

        if score > 0:
            scored_chunks.append((score, chunk))

    # Sort by score desc
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored_chunks[:top_k]]


# --- 3. CACHED MINER ---
def get_constraints_for_chunk(chunk, cache):
    # Create ID for cache
    c_id = hashlib.md5(chunk['chunk_text'].encode()).hexdigest()

    if c_id in cache:
        return cache[c_id]

    # Not in cache? Mine it!
    print(f"   ⛏️  Mining fresh chunk: {c_id[:8]}...")
    try:
        constraints = mine_constraints_from_chunk(chunk['chunk_text'])
        # If None (error), return empty list but don't cache it so we retry
        if constraints is None:
            return []

            # Save to cache
        cache[c_id] = constraints
        save_cache(cache)  # Commit immediately
        return constraints
    except Exception as e:
        print(f"Error mining: {e}")
        return []


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


# --- 4. THE JUDGE (LLM) ---
def evaluate_consistency(claim, constraints):
    if not constraints:
        # No evidence found in book? Default to Consistent (Innocent until proven guilty)
        return "consistent", "No constraints found regarding this topic."

    # Prepare Prompt
    evidence_text = "\n".join([f"- {c['constraint_text']} (Source: {c['source_excerpt']})" for c in constraints])

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

    resp = call_llm(prompt)
    try:
        # Clean and parse
        if "```" in resp: resp = resp.split("```")[1].replace("json", "")
        data = json.loads(resp)
        return data.get("verdict", "consistent"), data.get("reason", "")
    except:
        return "consistent", "Error parsing judge response"


# --- MAIN PIPELINE ---
def run_jit_pipeline():
    # Load Data
    chunks = load_chunks()
    cache = load_cache()

    with open(CLAIMS_FILE, 'r') as f:
        claims_data = json.load(f)  # Assuming list of objects

    results = []

    print(f"Processing {len(claims_data)} user claims...")

    for i, item in enumerate(claims_data):
        input_text = item.get('input_text', '')
        char_claims = item.get('claims', {})

        # Flatten all atomic claims into one text for search
        # Or you can loop through atomic claims. Let's do the whole input text for context.
        print(f"\n[{i + 1}/{len(claims_data)}] Claim: {input_text[:50]}...")

        # 1. RETRIEVE
        relevant_chunks = find_relevant_chunks(input_text, chunks)
        print(f"   🔍 Found {len(relevant_chunks)} relevant chunks.")

        # 2. MINE (JIT)
        all_constraints = []
        for ch in relevant_chunks:
            cons = get_constraints_for_chunk(ch, cache)
            all_constraints.extend(cons)

        # 3. JUDGE
        verdict, reason = evaluate_consistency(input_text, all_constraints)
        print(f"   ⚖️  Verdict: {verdict.upper()} ({reason})")

        # 4. SAVE
        results.append({
            "id": item.get("id"),
            "input_text": input_text,
            "verdict": verdict,
            "reason": reason,
            "evidence_used": all_constraints
        })

        # Simple Rate Limit sleep
        time.sleep(2)

        # Dump final
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_jit_pipeline()