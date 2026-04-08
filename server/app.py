from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import SaaSEnv
from models import Action

# ---------------------------------------------------------------------------
# App + global environment instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SaaS RL Environment API",
    description="OpenEnv-compatible SaaS Unit Economics Optimization Environment",
    version="1.0.0",
)

env = SaaSEnv()

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    action: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    """Reset the environment for the given task and return the initial observation."""
    valid_tasks = {"easy", "medium", "hard"}

    task = request.task if request else "easy"

    if task not in valid_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task '{task}'. Must be one of: {sorted(valid_tasks)}",
        )

    observation = env.reset(task=task)

    return {
        "observation": observation.model_dump(),
        "info": {}
    }


@app.post("/step")
def step(request: StepRequest):
    """Apply an action and return the resulting observation, reward, done flag, and info."""
    valid_actions = {a.value for a in Action}
    if request.action not in valid_actions:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{request.action}'. Must be one of: {sorted(valid_actions)}",
        )

    action = Action(request.action)
    result = env.step(action)

    return {
        "observation": result.observation.model_dump(),
        "reward":      result.reward,
        "done":        result.done,
        "info":        result.info,
    }


@app.get("/state")
def state():
    """Return the current observation without advancing the environment."""
    return {"observation": env.state().model_dump()}


def main():
    """Main entry point for the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


# Make app available for module import
__all__ = ["app"]


if __name__ == "__main__":
    main()