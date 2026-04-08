from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(Enum):
    run_ads               = "run_ads"
    improve_product       = "improve_product"
    run_retention_campaign = "run_retention_campaign"
    adjust_pricing        = "adjust_pricing"
    do_nothing            = "do_nothing"


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    users: int
    mrr: float
    budget: float
    churn_rate: float
    cac: float
    product_quality: float
    timestep: int


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)