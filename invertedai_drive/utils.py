import requests
import os
from dotenv import load_dotenv


class Client:
    def __init__(self):
        load_dotenv()
        self.dev = "DEV" in os.environ
        if not self.dev:
            self._endpoint = "https://api.inverted.ai/drive"
        else:
            self._endpoint = "http://localhost:8000"

    def run(self, api_key: str, model_inputs: dict) -> dict:
        response = requests.post(
            f"{self._endpoint}/drive",
            json=model_inputs,
            headers={
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "x-api-key": api_key,
                "api-key": api_key,
            },
        )
        return response.json()

    def initialize(
        self, api_key, location, agent_count=10, batch_size=1, min_speed=1, max_speed=3
    ):
        response = requests.get(
            f"{self._endpoint}/initialize",
            params={
                "location": location,
                "num_agents_to_spawn": agent_count,
                "num_samples": batch_size,
                "spawn_min_speed": min_speed,
                "spawn_max_speed": max_speed,
            },
            headers={
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "x-api-key": api_key,
                "api-key": api_key,
            },
        )

        return response.json()
