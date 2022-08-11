from requests.api import request
from invertedai_drive.utils import Client
from dataclasses import dataclass
import torch
import numpy as np
from typing import List, Union, Optional
import time

TIMEOUT = 10

InputDataType = Union[torch.Tensor, np.ndarray, List]


@dataclass
class Config:
    api_key: str
    location: str
    agent_count: int = 10
    batch_size: int = 1
    obs_length: int = 1
    step_times: int = 1
    min_speed: int = 10
    max_speed: int = 20


class Drive:
    def __init__(self, config) -> None:
        self.location = config.location
        self.config = config
        self.client = Client(self.config.api_key)

    def initialize(
        self,
        location=None,
        agent_count=None,
        batch_size=None,
        min_speed=None,
        max_speed=None,
    ) -> dict:
        start = time.time()
        timeout = TIMEOUT

        while True:
            try:
                initial_states = self.client.initialize(
                    location=location or self.config.location,
                    agent_count=agent_count or self.config.agent_count,
                    batch_size=batch_size or self.config.batch_size,
                    min_speed=min_speed or self.config.min_speed,
                    max_speed=max_speed or self.config.max_speed,
                )
                response = {
                    "states": initial_states["initial_condition"]["agent_states"],
                    "recurrent_states": None,
                    "attributes": initial_states["initial_condition"]["agent_sizes"],
                }
                return response
            except Exception as e:
                # TODO: Add logger
                print("Retrying")
                if timeout is not None and time.time() > start + timeout:
                    raise e

    def run(
        self,
        location: str,
        states: dict,
        agent_attributes: dict,
        recurrent_states: Optional[InputDataType] = None,
        present_masks: Optional[InputDataType] = None,
        return_birdviews: bool = False,
    ) -> dict:
        def _validate(input_dict: dict, input_name: str):
            input_data = input_dict[input_name]
            if isinstance(input_data, list):
                input_data = torch.Tensor(input_data)
            if input_data.shape[0] != self.config.batch_size:
                raise Exception(f"{input_name} has the wrong batch size (dim 0)")
            if input_data.shape[1] != self.config.agent_count:
                raise Exception(f"{input_name} has the wrong agent counts (dim 1)")
            if len(input_data.shape) > 2:
                # TODO: We hide the time dimension of the present masks for the client for now
                pass
            return input_data

        def _validate_recurrent_states(input_data: InputDataType):
            if isinstance(input_data, list):
                input_data = torch.Tensor(input_data)
            if input_data.shape[0] != self.config.batch_size:
                raise Exception("Recurrent states has the wrong batch size (dim 0)")
            if input_data.shape[1] != self.config.agent_count:
                raise Exception("Recurrent states has the wrong agent counts (dim 2)")
            if input_data.shape[2] != 2:
                raise Exception(
                    "Recurrent states has the wrong number of layers (dim 4)"
                )
            if input_data.shape[3] != 64:
                raise Exception("Recurrent states has the wrong dimension (dim 5)")
            return input_data

        def _tolist(input_data: InputDataType):
            if not isinstance(input_data, list):
                return input_data.tolist()
            else:
                return input_data

        def _validate_and_tolist(input_data: dict, input_name: str):
            return _tolist(_validate(input_data, input_name))

        present_masks = (
            _validate_and_tolist(present_masks, "present_masks")
            if present_masks is not None
            else None
        )  # BxA
        recurrent_states = (
            _tolist(_validate_recurrent_states(recurrent_states))
            if recurrent_states is not None
            else None
        )  # Bx(num_predictions)xAxTx2x64

        model_inputs = dict(
            location=location,
            initial_conditions=dict(
                agent_states=states,
                agent_sizes=agent_attributes,
            ),
            recurrent_states=recurrent_states,
            # Expand from BxA to BxAxT_total for the API interface
            present_masks=[
                [
                    [a for _ in range(self.config.obs_length + self.config.step_times)]
                    for a in b
                ]
                for b in present_masks
            ]
            if present_masks
            else None,
            batch_size=self.config.batch_size,
            agent_counts=self.config.agent_count,
            obs_length=self.config.obs_length,
            step_times=self.config.step_times,
            return_birdviews=return_birdviews,
        )

        start = time.time()
        timeout = TIMEOUT

        while True:
            try:
                return self.client.run(model_inputs)
            except Exception as e:
                # TODO: Add logger
                print("Retrying")
                if timeout is not None and time.time() > start + timeout:
                    raise e
