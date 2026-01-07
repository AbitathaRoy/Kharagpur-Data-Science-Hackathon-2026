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

    print(f"\n--- DEBUG START: '{caption}' ---")

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

    llm_response = call_llm(prompt)

    try:
        data = json.loads(llm_response)
    except json.JSONDecodeError:
        data = {"events": [], "traits": [], "causal_links": [], "assumptions": []}

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
    result= extract_caption_claims(
        "He had a difficult childhood."
        )
    print(result)