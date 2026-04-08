import requests
import os


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")


class SaaSEnvClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def reset(self, task: str = "easy"):
        response = requests.post(f"{self.base_url}/reset", json={"task": task})
        response.raise_for_status()
        return response.json()

    def step(self, action: str):
        response = requests.post(f"{self.base_url}/step", json={"action": action})
        response.raise_for_status()
        return response.json()

    def state(self):
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()