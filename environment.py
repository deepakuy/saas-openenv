from models import Observation, StepResult, Action


class SaaSEnv:
    def __init__(self):
        self.state_data = None

    def reset(self, task: str = "easy") -> Observation:
        if task == "easy":
            self.state_data = {
                "users": 50,
                "mrr": 2000.0,
                "budget": 30000.0,
                "churn_rate": 0.05,
                "cac": 50.0,
                "product_quality": 0.5,
                "timestep": 0,
            }

        elif task == "medium":
            self.state_data = {
                "users": 80,
                "mrr": 4000.0,
                "budget": 25000.0,
                "churn_rate": 0.12,
                "cac": 70.0,
                "product_quality": 0.4,
                "timestep": 0,
            }

        elif task == "hard":
            self.state_data = {
                "users": 60,
                "mrr": 3000.0,
                "budget": 15000.0,
                "churn_rate": 0.15,
                "cac": 80.0,
                "product_quality": 0.3,
                "timestep": 0,
            }

        return Observation(**self.state_data)

    def step(self, action: Action) -> StepResult:
        s = self.state_data

        reward = 0.0

        # --- ACTION EFFECTS ---
        if action == Action.run_ads:
            acquired = int(s["budget"] / s["cac"] * 0.1)
            s["users"] += acquired
            s["budget"] -= 2000
            reward += 5

        elif action == Action.improve_product:
            s["product_quality"] += 0.05
            s["churn_rate"] -= 0.01
            s["budget"] -= 1500
            reward += 8

        elif action == Action.run_retention_campaign:
            s["churn_rate"] -= 0.02
            s["budget"] -= 1000
            reward += 6

        elif action == Action.adjust_pricing:
            s["mrr"] *= 1.1
            s["churn_rate"] += 0.01
            reward += 4

        elif action == Action.do_nothing:
            reward -= 2

        # --- NATURAL DYNAMICS ---
        churned = int(s["users"] * s["churn_rate"])
        s["users"] -= churned
        s["mrr"] = s["users"] * 50

        # --- CLIP VALUES ---
        s["product_quality"] = max(0, min(1, s["product_quality"]))
        s["churn_rate"] = max(0, min(0.5, s["churn_rate"]))

        s["timestep"] += 1

        done = s["timestep"] >= 20 or s["users"] <= 0 or s["budget"] <= 0

        return StepResult(
            observation=Observation(**s),
            reward=reward,
            done=done,
            info={}
        )

    def state(self) -> Observation:
        return Observation(**self.state_data)