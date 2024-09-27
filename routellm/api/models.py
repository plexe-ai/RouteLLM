from pydantic import BaseModel
from enum import Enum

class RouterType(str, Enum):
    MF = "mf"
    CAUSAL_LLM = "causal_llm"
    RANDOM = "random"

from pydantic import BaseModel, field_validator, ValidationInfo

class RoutingPreference(BaseModel):
    accuracy: float
    cost: float
    speed: float
    threshold: float

    @field_validator('accuracy', 'cost', 'speed')
    def check_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Values must be between 0 and 1')
        return v

    @field_validator('speed')
    def check_sum(cls, v, info: ValidationInfo):
        values = info.data
        if 'accuracy' in values and 'cost' in values:
            total = values['accuracy'] + values['cost'] + v
            if not 0.99 <= total <= 1.01:  # Allow for small float precision errors
                raise ValueError('Sum of accuracy, cost, and speed must be 1')
        return v

class RoutingRequest(BaseModel):
    prompt: str
    preferences: RoutingPreference
