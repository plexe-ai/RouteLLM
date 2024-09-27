from fastapi import FastAPI, HTTPException
from .models import RoutingRequest, RouterType
from .router_selector import select_router, estimate_cost
from routellm.controller import Controller
import logging
import os

app = FastAPI()

enabled_routers = os.getenv('ENABLED_ROUTERS', 'mf').split(',')

controller = Controller(
    routers=enabled_routers,
    strong_model="openai/gpt-4-1106-preview",
    weak_model="together_ai/togethercomputer/Llama-2-7B-32K-Instruct"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/route")
async def route_request(request: RoutingRequest):
    router_type, initial_model = select_router(request.preferences)
    logger.info(f"Selected router: {router_type}, Initial model: {initial_model}")
    
    if router_type is None:
        model_name = controller.model_pair.strong
    else:
        if router_type not in controller.routers:
            raise HTTPException(status_code=400, detail=f"Router {router_type} not available")
        model_name = f"router-{router_type}-{request.preferences.threshold}"
    
    logger.info(f"Model name for controller: {model_name}")

    try:
        response = await controller.acompletion(
            model=model_name,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=0.7,
            max_tokens=150
        )
        llm_response = response.choices[0].message.content
        
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        # Determine the actual model used
        if router_type is None:
            routed_model = model_name  # This is a direct model call
        else:
            routed_model = response.model  # This is the model selected by the router
        
        logger.info(f"Actual model used: {routed_model}")
        
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {str(e)}")

    estimated_cost = estimate_cost(router_type, routed_model)
    
    return {
        "router": router_type if router_type else "direct",
        "model": routed_model,
        "estimated_cost": estimated_cost,
        "llm_response": llm_response,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }