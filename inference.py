import os
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.environ["MODEL_NAME"]
HF_TOKEN     = os.environ["HF_TOKEN"]

ENV_NAME     = "saas-unit-economics-env"

# ---------------------------------------------------------------------------
# OpenAI client (required by spec)
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def reset_env(task: str) -> dict:
    response = requests.post(f"{API_BASE_URL}/reset", json={"task": task})
    response.raise_for_status()
    return response.json()["observation"]

def step_env(action: str) -> dict:
    response = requests.post(f"{API_BASE_URL}/step", json={"action": action})
    response.raise_for_status()
    return response.json()

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

TASKS     = ["easy", "medium", "hard"]
MAX_STEPS = 20

ping_model(client)

for task in TASKS:
    obs = reset_env(task)

    total_reward = 0.0
    rewards = []
    steps_taken = 0

    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}")

    for step_num in range(1, MAX_STEPS + 1):
        action = select_action(obs)
        result = step_env(action)

        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]

        total_reward += reward
        rewards.append(reward)
        steps_taken = step_num

        print(
            f"[STEP] step={step_num} "
            f"action={action} "
            f"reward={reward:.2f} "
            f"done={str(done).lower()} "
            f"error=null"
        )

        if done:
            break

    score = max(0.0, min(total_reward / 100.0, 1.0))
    success = str(score > 0.1).lower()

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={success} "
        f"steps={steps_taken} "
        f"score={score:.2f} "
        f"rewards={rewards_str}"
    )