# schemas/claims.py

from dataclasses import dataclass
from typing import List, Literal

TimeScope = Literal["childhood", "adolescence", "adulthood", "unspecified"]
TraitScope = Literal["stable", "acquired", "temporary"]
CausalConfidence = Literal["explicit", "implied"]

def normalize_time_scope(value):
    allowed = {"childhood", "adolescence", "adulthood", "unspecified"}
    return value if value in allowed else "unspecified"

@dataclass
class EventClaim:
    description: str
    time_scope: TimeScope

@dataclass
class TraitClaim:
    trait: str
    time_scope: TraitScope

@dataclass
class CausalLink:
    cause: str
    effect: str
    confidence: CausalConfidence

@dataclass
class CaptionClaims:
    events: List[EventClaim]
    traits: List[TraitClaim]
    causal_links: List[CausalLink]
    assumptions: List[str]
