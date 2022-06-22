import requests
import time
import os
import json
from uuid import uuid4
import torch
from invertedai_drive.models import ModelOutputs
from dotenv import load_dotenv


class Client:
    def __init__(self):
        load_dotenv()
        self.dev = 'HOST' in os.environ
        self._endpoint = (
            "https://api.banana.dev/" if not self.dev else os.environ.get('HOST')
        )

    def run(self, api_key: str, model_key: str, model_inputs: dict) -> ModelOutputs:
        if self.dev:
            response = requests.post(self._endpoint, json=model_inputs)
            return response.json()
        else:
            result = self._start(api_key, model_key, model_inputs)

            def _extract_model_outputs(dict_model_output):
                states = torch.Tensor(dict_model_output["states"])
                recurrent_states = torch.Tensor(dict_model_output["recurrent_states"])
                model_outputs = ModelOutputs(states, recurrent_states)
                return model_outputs

            if result["finished"]:
                return _extract_model_outputs(result["modelOutputs"])

            # If it's long running, so poll for result
            while True:
                dict_out = self._check(api_key, result["callID"])
                if dict_out["message"].lower() == "success":
                    return _extract_model_outputs(result["modelOutputs"])

    def _start(self, api_key, model_key, model_inputs, start_only=False):
        route_start = "start/v3/"
        url_start = self._endpoint + route_start

        payload = {
            "id": str(uuid4()),
            "created": int(time.time()),
            "apiKey": api_key,
            "modelKey": model_key,
            "modelInputs": model_inputs,
            "startOnly": start_only,
        }

        response = requests.post(url_start, json=payload)

        if response.status_code != 200:
            raise Exception(f"server error: status code {response.status_code}")

        try:
            out = response.json()
        except:
            raise Exception("server error: returned invalid json")

        if "error" in out["message"].lower():
            raise Exception(out["message"])

        return out

    def _check(self, api_key, call_id):
        route_check = "check/v3/"
        url_check = self._endpoint + route_check
        # Poll server for completed task

        payload = {
            "id": str(uuid4()),
            "created": int(time.time()),
            "longPoll": True,
            "callID": call_id,
            "apiKey": api_key,
        }
        response = requests.post(url_check, json=payload)

        if response.status_code != 200:
            raise Exception(f"server error: status code {response.status_code}")

        try:
            out = response.json()
        except:
            raise Exception("server error: returned invalid json")

        try:
            if "error" in out["message"].lower():
                raise Exception(out["message"])
            return out
        except Exception as e:
            raise e
