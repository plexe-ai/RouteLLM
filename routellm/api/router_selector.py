from .models import RouterType, RoutingPreference
from typing import Optional, Tuple

def select_router(preferences: RoutingPreference) -> Tuple[RouterType, str]:

    direct_strong_score = preferences.accuracy
    causal_llm_score = (preferences.accuracy * 0.7 + preferences.cost * 0.2 + preferences.speed * 0.1)
    mf_score = (preferences.accuracy * 0.3 + preferences.cost * 0.3 + preferences.speed * 0.4)

    max_score = max(direct_strong_score, causal_llm_score, mf_score)

    if max_score == direct_strong_score:
        return None, "strong_model"
    elif max_score == causal_llm_score:
        return RouterType.CAUSAL_LLM, "weak_model"
    else:
        return RouterType.MF.value, "weak_model"

def estimate_cost(router_type: Optional[RouterType], model: str) -> float:
    #TODO: Verify actual tokens 
    costs = {
        "openai/gpt-4-1106-preview": 0.01,
        "together_ai/togethercomputer/Llama-2-7B-32K-Instruct": 0.001
    }
    router_overhead = {
        RouterType.MF: 0.0001,
        RouterType.CAUSAL_LLM: 0.001,
        RouterType.RANDOM: 0.00001
    }
    
    # Default cost if the model is not in the costs dictionary
    model_cost = costs.get(model, 0.005)  # Assuming a default cost of 0.005 for unknown models
    
    # If router_type is None (direct call), don't add router overhead
    if router_type is None:
        return model_cost
    
    # Get router overhead, default to 0 if router type is not in the dictionary
    overhead = router_overhead.get(router_type, 0)
    
    return model_cost + overhead