from __future__ import annotations

def task_family(user_request: str) -> str:
    """Analyze the user request and determine the task family based on keywords."""
    req = user_request.lower()
    
    if "fibonacci" in req:
        return "fibonacci_dp"
    elif "sort" in req:
        return "sort"
    elif "actor_critic" in req or "actor-critic" in req:
        return "actor_critic"
    elif "filter" in req or "moving average" in req:
        return "signal_filter"
    elif "linear system" in req or "numerically" in req or "numerical" in req:
        return "numerical"
        
    return "generic"


def build_context_key(*, user_request: str, model_name: str, has_baseline: bool) -> str:
    """Build a deterministic placeholder context key."""
    family = task_family(user_request)
    
    mode = "edit" if has_baseline else "new"
    safe_model = (model_name or "unknown").strip().lower() or "unknown"
    
    return f"{family}|{safe_model}|{mode}"


__all__ = ["build_context_key", "task_family"]