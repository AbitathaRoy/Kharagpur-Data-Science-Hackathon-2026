# extraction/constraint_miner.py

import json
from utils.llm_client import call_llm

# --- THE EXACT PROMPT FROM CHATGPT ---
CONSTRAINT_PROMPT_TEMPLATE = """
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

If the passage contains NO such statements, return an empty list.

---

OUTPUT FORMAT (STRICT JSON):

[
  {{
    "character": "string",
    "constraint_type": "hard_fact | temporal_lock | causal_allocation | invariant_trait",
    "constraint_text": "plain language description of the constraint",
    "time_scope": "childhood | adulthood | specific time | lifespan | unspecified",
    "source_excerpt": "verbatim quote from the passage"
  }}
]

---

PASSAGE:
<<<
{chunk_text}
>>>
"""

def mine_constraints_from_chunk(chunk_text: str) -> list[dict]:
    """
    UDF (User Defined Function) for Pathway.
    Takes a text chunk, runs the LLM, returns a list of constraint dicts.
    """
    if not chunk_text or len(chunk_text) < 50:
        return []

    prompt = CONSTRAINT_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    
    # Call your existing utility
    response_str = call_llm(prompt) # model="gpt-4o" Or "llama-3.3-70b-versatile"
    
    try:
        # The prompt asks for a JSON list directly.
        data = json.loads(response_str)
        
        # Basic validation: ensure it's a list
        if isinstance(data, dict) and "constraints" in data:
            return data["constraints"]
        if isinstance(data, list):
            return data
            
        return []
    except Exception as e:
        print(f"Error parsing constraint JSON: {e}")
        return []