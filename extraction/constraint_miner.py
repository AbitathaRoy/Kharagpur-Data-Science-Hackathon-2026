# extraction/constraint_miner.py

import json
import re  # <--- NEW: For regex cleaning
from utils.llm_client import call_llm

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

If the passage contains NO such statements, return:
{ "constraints": [] }

---

OUTPUT FORMAT (STRICT JSON):

{
  "constraints": [
    {
      "character": "string",
      "constraint_type": "hard_fact | temporal_lock | causal_allocation | invariant_trait",
      "constraint_text": "plain language description of the constraint",
      "time_scope": "childhood | adulthood | specific time | lifespan | unspecified",
      "source_excerpt": "verbatim quote from the passage"
    }
  ]
}


---

PASSAGE:
<<<
{chunk_text}
>>>
"""

def clean_json_string(raw_text: str) -> str:
    """
    Attempts to extract valid JSON from a messy LLM response.
    Removes markdown fences (```json ... ```) and preambles.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # 1. Remove Markdown Code Blocks if present
    if "```" in text:
        # Regex to capture content inside ```json ... ``` or just ``` ... ```
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    return text


# extraction/constraint_miner.py

def mine_constraints_from_chunk(chunk_text: str) -> list[dict] | None:
    if not chunk_text or len(chunk_text) < 50:
        return []

    prompt = CONSTRAINT_PROMPT_TEMPLATE.format(chunk_text=chunk_text)

    try:
        raw_response = call_llm(prompt)
        cleaned_response = clean_json_string(raw_response)

        if not cleaned_response:
            return None

        data = json.loads(cleaned_response)

        # ✅ HARDENED SCHEMA HANDLING
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if "constraints" in data and isinstance(data["constraints"], list):
                return data["constraints"]
            # allow empty dict → empty constraints
            return []

        return []

    except json.JSONDecodeError:
        print("⚠️ JSON Parse Error.")
        return None

    except Exception as e:
        print(f"⚠️ Miner exception: {e}")
        return None
