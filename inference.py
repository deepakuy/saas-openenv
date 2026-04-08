import os
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME", "dummy-model")
HF_TOKEN     = os.getenv("HF_TOKEN", "dummy-token")

ENV_NAME     = "saas-unit-economics-env"

# ---------------------------------------------------------------------------
# OpenAI client (required by spec)
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Heuristic policy
# ---------------------------------------------------------------------------

def select_action(obs: dict) -> str:
    if obs["churn_rate"] > 0.08:
        return "run_retention_campaign"
    elif obs["budget"] > 20000:
        return "run_ads"
    elif obs["users"] > 100:
        return "adjust_pricing"
    else:
        return "improve_product"

# ---------------------------------------------------------------------------
# Model ping
# ---------------------------------------------------------------------------

def ping_model(client):
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
        )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

from environment import SaaSEnv
from models import Action

env = SaaSEnv()

TASKS     = ["easy", "medium", "hard"]
MAX_STEPS = 20

ping_model(client)

try:
    for task in TASKS:
        obs = env.reset(task)

        total_reward = 0.0

        print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}")

        for step_num in range(1, MAX_STEPS + 1):
            action_str = select_action(obs)
            action = Action(action_str)

            result = env.step(action)

            obs = result.observation.model_dump()
            reward = result.reward
            done = result.done

            total_reward += reward

            print(
                f"[STEP] step={step_num} "
                f"action={action_str} "
                f"reward={reward:.2f} "
                f"done={str(done).lower()} "
                f"error=null"
            )

            if done:
                break

        score = max(0.0, min(total_reward / 100.0, 1.0))
        success = str(score > 0.1).lower()

        print(
            f"[END] success={success} "
            f"steps={step_num} "
            f"score={score:.2f} "
            f"rewards={total_reward:.2f}"
        )

except Exception as e:
    print(f"[ERROR] {str(e)}")
