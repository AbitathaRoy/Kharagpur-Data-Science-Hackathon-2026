# extraction/caption_claims.py

import json
import sys
import os
from typing import List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from schemas.claims import CaptionClaims, EventClaim, TraitClaim, CausalLink
from utils.llm_client import call_llm

# -------------------------
# Caching Configuration
# -------------------------
CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'caption_claims_cache.json')

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_cache_entry(caption, data):
    # Reload to ensure we don't overwrite concurrent runs (simple approach)
    current_cache = load_cache()
    current_cache[caption] = data
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(current_cache, f, indent=2)

# Load cache globally on startup
GLOBAL_CACHE = load_cache()


# -------------------------
# Hard-coded guards (DO NOT REMOVE)
# -------------------------

CAUSAL_TRIGGERS = [
    "because", "caused", "causes", "made", "led to",
    "shaped", "as a result", "resulted in", "due to"
]

EVALUATIVE_WORDS = [
    "difficult", "troubled", "harsh", "hard", "traumatic", "painful"
]

LIFE_PERIOD_TERMS = [
    "childhood", "upbringing", "past", "youth", "life"
]

VALID_EVENT_TIMES = {"childhood", "adolescence", "adulthood", "unspecified"}
VALID_TRAIT_TIMES = {"stable", "acquired", "temporary"}
VALID_CAUSAL_CONF = {"explicit", "implied"}


# -------------------------
# Helper functions
# -------------------------

def caption_has_explicit_causality(caption: str) -> bool:
    c = caption.lower()
    return any(trigger in c for trigger in CAUSAL_TRIGGERS)


def is_life_period_trait(trait: str) -> bool:
    t = trait.lower()
    return any(term in t for term in LIFE_PERIOD_TERMS)


def normalize(value: str, allowed: set, default: str):
    return value if value in allowed else default


# -------------------------
# Main extractor
# -------------------------

def extract_caption_claims(caption: str) -> CaptionClaims:
    """
    Phase 2 (FINAL):
    Expand a hypothetical backstory caption into atomic claims.
    Deterministic guards enforce logical discipline.
    """

    print(f"\n--- DEBUG START: '{caption[:50]}...' ---")

    # --- CACHE CHECK ---
    # Logic: If data exists in cache, use it. 
    # If not, call LLM.
    # If LLM fails, DO NOT SAVE to cache (so it retries next time).
    
    data = None
    
    if caption in GLOBAL_CACHE:
        print("   ⚡ CACHE HIT: Using saved extraction.")
        data = GLOBAL_CACHE[caption]
    else:
        print("   ⛏️  CACHE MISS: Calling LLM...")
        
        prompt = f"""
You are extracting atomic claims from a hypothetical character backstory.

CRITICAL CONSTRAINTS (VIOLATIONS ARE ERRORS):
- Do NOT use real-world knowledge, plausibility, or common sense.
- Do NOT infer causality from sequence, geography, or context.
- ONLY extract causal_links if the caption itself asserts causation
  using explicit causal language (e.g., "made", "led to", "because").
- Life periods or circumstances (e.g., "difficult childhood") are EVENTS,
  not personality traits.
- If the caption uses vague evaluative language, you MUST list the hidden assumptions.

IMPORTANT RULES:
- Do NOT judge whether the backstory is true.
- Only extract claims that MUST be true if the backstory were correct.
- Always choose the most specific time_scope possible.
- Output MUST be valid JSON matching the schema exactly.
- Use empty lists instead of null.

JSON SCHEMA (FOLLOW EXACTLY):
{{
  "events": [
    {{
      "description": "string",
      "time_scope": "childhood | adolescence | adulthood | unspecified"
    }}
  ],
  "traits": [
    {{
      "trait": "string",
      "time_scope": "stable | acquired | temporary"
    }}
  ],
  "causal_links": [
    {{
      "cause": "string",
      "effect": "string",
      "confidence": "explicit | implied"
    }}
  ],
  "assumptions": [
    "string"
  ]
}}

BACKSTORY CAPTION:
"{caption}"
"""
        
        try:
            llm_response = call_llm(prompt)
            # Try parsing
            parsed_data = json.loads(llm_response)
            
            # If successful, assign to data AND save to cache
            data = parsed_data
            save_cache_entry(caption, data)
            # Update global variable to avoid reload lag
            GLOBAL_CACHE[caption] = data
            
        except json.JSONDecodeError:
            print("   ⚠️  ERROR: JSON Parse Failed. Will retry next run.")
            # Set default for this run so it doesn't crash, 
            # BUT DO NOT SAVE TO CACHE.
            data = {"events": [], "traits": [], "causal_links": [], "assumptions": []}
        except Exception as e:
            print(f"   ⚠️  ERROR: API Failure ({e}). Will retry next run.")
            data = {"events": [], "traits": [], "causal_links": [], "assumptions": []}

    # Ensure data is not None (fallback)
    if not data:
        data = {"events": [], "traits": [], "causal_links": [], "assumptions": []}

    # ---------------------------------------------------------
    # Processing Logic (Runs on both Cached and New data)
    # ---------------------------------------------------------
    
    # 2. DEBUG: Check Causality Trigger
    explicit_causality = caption_has_explicit_causality(caption)
    print(f"DEBUG: Explicit Causality Detected? -> {explicit_causality}")
    if explicit_causality:
        print(f"DEBUG: Trigger word found: {[t for t in CAUSAL_TRIGGERS if t in caption.lower()]}")

    # ---- EVENTS ----
    events = []
    for e in data.get("events", []):
        events.append(
            EventClaim(
                description=e.get("description", "").strip(),
                time_scope=normalize(
                    e.get("time_scope", "unspecified"),
                    VALID_EVENT_TIMES,
                    "unspecified"
                )
            )
        )

    # ---- TRAITS (HARD FILTER) ----
    # 3. DEBUG: Check Trait Filtering
    traits = []
    for t in data.get("traits", []):
        trait_name = t.get("trait", "").strip()
        is_period = is_life_period_trait(trait_name)
        
        print(f"DEBUG: Checking Trait: '{trait_name}'")
        print(f"       Is Life Period? -> {is_period}")
        
        if not is_period:
            traits.append(TraitClaim(
                trait=trait_name,
                time_scope=normalize(t.get("time_scope", "acquired"), VALID_TRAIT_TIMES, "acquired")
            ))
        else:
            print(f"       BLOCKED: '{trait_name}' removed by guard.")

    # ---- CAUSAL LINKS (ALL OR NOTHING) ----
    causal_links = []
    if explicit_causality:
        for c in data.get("causal_links", []):
            causal_links.append(
                CausalLink(
                    cause=c.get("cause", "").strip(),
                    effect=c.get("effect", "").strip(),
                    confidence=normalize(
                        c.get("confidence", "implied"),
                        VALID_CAUSAL_CONF,
                        "implied"
                    )
                )
            )
    # else: causal_links stays []

    # ---- ASSUMPTIONS (FORCED) ----
    assumptions = data.get("assumptions", []) or []

    if any(word in caption.lower() for word in EVALUATIVE_WORDS):
        if not assumptions:
            assumptions = [
                "the described condition was sustained rather than temporary",
                "the described condition had psychological or developmental significance"
            ]

    print("--- DEBUG END ---\n")

    return CaptionClaims(
        events=events,
        traits=traits,
        causal_links=causal_links,
        assumptions=assumptions
    )


if __name__ == "__main__":
    result = extract_caption_claims(
        "He had a difficult childhood."
        )
    print(result)