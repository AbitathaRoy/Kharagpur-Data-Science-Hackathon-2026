# schemas/constraints.py
# for novel

from pydantic import BaseModel, Field
from typing import Literal, Optional, List

class WorldConstraint(BaseModel):
    character: str = Field(..., description="Standardized name of the character")
    constraint_type: Literal["hard_fact", "temporal_lock", "causal_allocation", "invariant_trait"]
    constraint_text: str = Field(..., description="The atomic fact or rule")
    time_scope: str = Field(..., description="childhood | adulthood | specific time | lifespan | unspecified")
    source_excerpt: str = Field(..., description="Verbatim quote from the text")

class ConstraintList(BaseModel):
    constraints: List[WorldConstraint]