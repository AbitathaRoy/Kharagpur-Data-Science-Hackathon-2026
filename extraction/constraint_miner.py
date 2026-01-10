# extraction/constraint_miner.py

import json
import re
from utils.llm_client import call_llm

def build_constraint_prompt(chunk_text):
    """
    Constructs the prompt using an f-string to avoid .format() collisions 
    with JSON curly braces.
    """
    return f"""
You are analyzing a passage from a novel.

Your task is to extract ONLY statements that behave like
RULES, CONSTRAINTS, or IRREVERSIBLE FACTS about a character.

You may extract a statement ONLY if it:
- restricts what could have happened in the past, OR
- fixes when something became true, OR
- explicitly assigns a cause to a trait, belief, or action, OR
- asserts a trait or belief persisted over a long period.

DO NOT extract:
- single events without lasting implications
- descriptions of scenes or emotions
- character development without explicit restriction
- plausible interpretations or inferences
- anything not explicitly stated in the text

DO NOT use real-world knowledge.
DO NOT infer beyond the passage.
DO NOT summarize the passage.

If the passage contains NO such statements, return an empty list inside the JSON.

---

OUTPUT FORMAT (STRICT JSON):
{{
  "constraints": [
    {{
      "character": "string",
      "constraint_type": "hard_fact | temporal_lock | causal_allocation | invariant_trait",
      "constraint_text": "plain language description of the constraint",
      "time_scope": "childhood | adulthood | specific time | lifespan | unspecified",
      "source_excerpt": "verbatim quote from the passage"
    }}
  ]
}}

PASSAGE:
<<<
{chunk_text}
>>>
"""

def clean_json_string(raw_text: str) -> str:
    """
    Aggressively extracts JSON object from messy LLM output.
    """
    if not raw_text:
        return ""
        
    text = raw_text.strip()
    
    # Strategy 1: Look for Markdown blocks
    if "```" in text:
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            
    # Strategy 2: Look for the outer-most curly braces {}
    # We now expect an OBJECT, not a LIST, because of JSON mode.
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
        
    return text

def mine_constraints_from_chunk(chunk_text: str) -> list[dict]:
    if not chunk_text or len(chunk_text) < 50:
        return []

    # Use the function to build prompt safely
    prompt = build_constraint_prompt(chunk_text)

    try:
        raw_response = call_llm(prompt)

        # 1. API FAILURE
        if raw_response is None:
            print("❌ [MINER] API returned None (Network/Rate Limit).")
            return None  # Triggers Retry

        # 2. CLEANING
        cleaned_response = clean_json_string(raw_response)
        
        if not cleaned_response:
            if "error" in raw_response.lower() or "sorry" in raw_response.lower():
                 print(f"⚠️ [MINER] Model Refusal: {raw_response[:50]}...")
            return []

        # 3. PARSING
        try:
            data = json.loads(cleaned_response)
            
            # We expect {"constraints": [...]}
            if isinstance(data, dict):
                return data.get("constraints", [])
                
            # Fallback if model output a list directly
            if isinstance(data, list):
                return data
                
            return [] 

        except json.JSONDecodeError:
            # 4. FALLBACK STRATEGY (The "Dirty" Fix)
            print(f"⚠️ [MINER] JSON Parse Failed. Saving RAW output.")
            return [{
                "character": "System_Fallback",
                "constraint_type": "hard_fact",
                "constraint_text": f"Raw Analysis (JSON Error): {cleaned_response}",
                "time_scope": "unspecified",
                "source_excerpt": "PARSE_FAILURE"
            }]

    except Exception as e:
        # 5. CATCH-ALL SAFETY NET
        print(f"⚠️ [MINER] Unexpected Error: {e}")
        return None