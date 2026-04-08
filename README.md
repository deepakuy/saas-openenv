---
title: SaaS Unit Economics Optimization Environment
emoji:  "🚀"
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---

# SaaS Unit Economics Optimization Environment


> An OpenEnv-compatible reinforcement learning environment where an AI agent manages a SaaS startup — balancing user growth, churn, CAC, and runway to maximise Monthly Recurring Revenue.

> Fully compliant with the OpenEnv specification (step/reset/state API, typed schema, and reproducible evaluation).

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://huggingface.co/openenv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/deploy-Docker-2496ED)](https://www.docker.com/)

---

## 1. Problem

Every SaaS company faces the same core tension: **grow fast or die slow**.

The decisions a growth manager makes in the first 24 months define whether the company survives. These decisions are non-trivial because:

- Spending on user acquisition raises CAC over time (diminishing returns on ads)
- Ignoring churn means a leaky bucket — growth never compounds
- Raising prices boosts revenue but spikes churn temporarily
- Product investment reduces churn slowly but costs runway today
- Doing nothing preserves cash but lets the user base decay

There is no single dominant strategy. The optimal sequence of decisions depends on the starting state, available budget, and current churn dynamics — which makes this a natural fit for reinforcement learning.

---

## 2. Solution

This environment simulates the unit economics of an early-stage SaaS company over a 24-month horizon. An RL agent acts as the growth manager, choosing one of five discrete actions per timestep (month).

The environment follows the **OpenEnv specification** and exposes a standard HTTP interface:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/reset` | Start a new episode for a given task |
| `POST` | `/step`  | Apply an action, receive next state + reward |
| `GET`  | `/state` | Read current observation without advancing |

Each episode terminates when the company goes bankrupt, loses all users, or completes 24 months.

---

## 3. State Space

The agent observes 7 variables at every timestep:

| Variable | Type | Description |
|----------|------|-------------|
| `users` | `int` | Number of active paying customers |
| `mrr` | `float` | Monthly Recurring Revenue in USD (`users × price_per_user`) |
| `budget` | `float` | Remaining operational cash in USD |
| `churn_rate` | `float` | Fraction of users lost per month. Floor: `0.02`, ceiling: `0.50` |
| `cac` | `float` | Customer Acquisition Cost in USD. Floor: `$50`, ceiling: `$500` |
| `product_quality` | `float` | Abstract product quality score `[0.0, 1.0]`. Higher values reduce churn |
| `timestep` | `int` | Current month index `[0, 24]` |

---

## 4. Action Space

Five discrete actions. Exactly one is taken per timestep.

### `run_ads`
Spends **$5,000** on paid acquisition. Acquires `floor(5000 / CAC)` new users. CAC increases by **15%** after each use (diminishing returns). CAC is capped at $500.

**Trade-off:** Fast user growth, but repeated use inflates CAC and drains budget without proportional return.

---

### `improve_product`
Invests **$3,000** in product development. Increases `product_quality` by `+0.1` (capped at `1.0`). Reduces `churn_rate` by a fixed `0.01`.

**Trade-off:** No immediate user growth. Slow, compounding payoff. Essential for long-run retention.

---

### `run_retention_campaign`
Spends **$2,000** on a targeted retention campaign. Immediately reduces `churn_rate` by `0.03`.

**Trade-off:** Fast, cheap churn fix. One-time effect per use — does not compound. Treats the symptom, not the cause. Overusing it without product investment is an expensive patch.

---

### `adjust_pricing`
No direct cost. Increases `price_per_user` by **10%**, raising MRR immediately. Triggers a one-time `+0.02` churn spike. Subject to a **4-step cooldown** between uses.

**Trade-off:** Pure revenue optimisation with no user growth. Best deployed when churn is already under control. Poor timing amplifies user loss.

---

### `do_nothing`
No spend. CAC decreases by **5%** (market cooldown). Natural churn still applies.

**Trade-off:** Preserves runway and lets CAC recover. However, `$4,000` in fixed operating costs is deducted every month regardless. Passive play is slow decay, not safety.

---

## 5. Reward Function

Reward is **dense** — issued at every timestep. This prevents sparse credit assignment across 24 steps.

```
reward = (mrr / 20.0)
       + (new_users × 0.1)
       - (users_lost × 0.2)
       - max(0, (cac - 150) / 100)
       - 2.0  [if budget < $10,000]
       - 50.0 [if budget ≤ 0, terminal]

reward = clip(reward, -10.0, 10.0)
```

| Component | Effect | Rationale |
|-----------|--------|-----------|
| `mrr / 20` | Positive | Primary signal: rewards revenue generation |
| `new_users × 0.1` | Positive | Secondary bonus for user acquisition |
| `users_lost × 0.2` | Negative | Churn costs more than acquisition gains — reflects LTV asymmetry |
| `max(0, (cac - 150) / 100)` | Negative | Penalises inefficient ad spend above the $150 threshold |
| `-2.0 if budget < $10K` | Negative | Early warning: runway is critically low |
| `-50.0 if budget ≤ 0` | Negative | Terminal bankruptcy penalty |
| `clip(−10, +10)` | Bounds | Prevents reward scale explosion across tasks |

---

## 6. Tasks

Three tasks with qualitatively different decision challenges — not just parameter scaling.

### Easy — *Stable Growth*

| Parameter | Value |
|-----------|-------|
| Users | 50 |
| Budget | $30,000 |
| Churn rate | 0.05 |
| CAC | $100 |
| Product quality | 0.5 |

**Challenge:** Basic budget management and growth pacing. Conditions are forgiving. The agent must learn not to spam `run_ads` and to alternate with product investment.

**Grader:** `1.0` → users ≥ 80 + MRR ≥ $4K · `0.7` → users ≥ 40 + MRR ≥ $2K · `0.4` → survived (users > 20) · `0.0` → bankrupt

---

### Medium — *Retention Crisis*

| Parameter | Value |
|-----------|-------|
| Users | 80 |
| Budget | $25,000 |
| Churn rate | 0.12 |
| CAC | $120 |
| Product quality | 0.3 |

**Challenge:** High starting churn and weak product quality. The agent must stabilise the leaky bucket before scaling acquisition. A naïve agent that only runs ads will watch the user base collapse.

**Grader:** `1.0` → users ≥ 70 + MRR ≥ $3.5K · `0.7` → users ≥ 40 + MRR ≥ $2K · `0.4` → survived (users > 20) · `0.0` → bankrupt

---

### Hard — *Tight Runway*

| Parameter | Value |
|-----------|-------|
| Users | 60 |
| Budget | $15,000 |
| Churn rate | 0.10 |
| CAC | $140 |
| Product quality | 0.4 |

**Challenge:** Tight cash reserve (~3 months of fixed costs), elevated CAC, and a required MRR growth target of **$8,000** by episode end. The agent cannot just conserve cash — it must grow. It cannot just spend on ads — it will go bankrupt. Every action must be justified.

**Grader:** `1.0` → MRR ≥ $8K + users ≥ 100 · `0.7` → MRR ≥ $8K (growth target met) · `0.4` → survived, MRR < $8K · `0.0` → bankrupt

---

## 7. How to Run

### Prerequisites
- Docker
- Python 3.10+

### Build and Run

```bash
# Build the Docker image
docker build -t saas-env .

# Run the server on port 7860
docker run -p 7860:7860 saas-env
```

The server starts at `http://localhost:7860`. Confirm it's running:

```bash
curl http://localhost:7860/state
```

### API Usage

**Reset to a task:**
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'
```

**Step with an action:**
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "run_ads"}'
```

**Read current state:**
```bash
curl http://localhost:7860/state
```

**Available actions:** `run_ads` · `improve_product` · `run_retention_campaign` · `adjust_pricing` · `do_nothing`

**Available tasks:** `easy` · `medium` · `hard`

### Run Inference Script

```bash
export API_BASE_URL="http://localhost:7860"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"

python inference.py
```

Expected output format:
```
[START] task=easy
[STEP] step=1 action=run_ads reward=3.45 done=False
[STEP] step=2 action=improve_product reward=2.10 done=False
...
[END] score=0.72
```

### Interactive API Docs

FastAPI auto-generates Swagger documentation at:
```
http://localhost:7860/docs
```

---

## 8. Why This Matters

SaaS unit economics is not an academic construct — it is the framework used by every growth team, investor, and operator evaluating whether a software business is viable. The decisions modelled in this environment (CAC vs LTV, churn management, pricing elasticity, runway allocation) are made daily by real companies managing real capital.

This environment encodes real SaaS benchmarks such as CAC efficiency thresholds and churn sensitivity, making it suitable for evaluating agent decision-making in business-critical scenarios.

This environment contributes three things:

**A learnable real-world decision problem.** The 5-action, 7-variable design is simple enough for agents to learn within compute budgets, yet complex enough that no single action dominates. The optimal policy requires sequencing — investing in product before scaling ads, timing price increases around stable churn — which distinguishes it from single-lever optimisation tasks.

**An interpretable evaluation framework.** Because the state variables are domain-legible (users, MRR, budget), the behaviour of a learned policy can be evaluated by domain experts without needing to understand RL. A policy that "invests in product early, then scales acquisition once churn is controlled" is a statement about business strategy, not just a score.

**A deployable benchmark.** The three task tiers create a natural difficulty progression that can benchmark agent learning curves, test generalisation across starting conditions, and evaluate how well agents recover from constrained initial states.

---

## Baseline Performance

Using a simple heuristic agent (prioritise retention when churn > 0.08, else acquire users, adjust pricing when users > 100, otherwise improve product):

| Task | Score Range |
|------|-------------|
| Easy | 0.8 – 1.0 |
| Medium | 0.6 – 0.8 |
| Hard | 0.4 – 0.7 |

This demonstrates the environment is learnable but non-trivial — a naïve policy performs well on Easy but degrades meaningfully on Medium and Hard, where correct action sequencing is required.

---

## Project Structure

```
├── models.py          # Action, Observation, StepResult typed models (Pydantic BaseModel)
├── environment.py     # SaaSEnv — core environment logic
├── server/app.py       # FastAPI server exposing /reset, /step, /state
├── inference.py       # End-to-end inference script with strict logging
├── openenv.yaml       # OpenEnv manifest
├── requirements.txt   # Python dependencies
└── Dockerfile         # Container definition for HF Spaces deployment
```

---

## Requirements

```
fastapi
uvicorn
pydantic
requests
openai
```

---

*Built for the Meta × Hugging Face × OpenEnv × Scaler Hackathon.*