from invertedai.error import TryAgain
from dataclasses import dataclass
import torch
import numpy as np
from typing import List, Union, Optional
import time
import invertedai as iai


TIMEOUT = 10

InputDataType = Union[torch.Tensor, np.ndarray, List]


@dataclass
class Config:
    api_key: str = ""
    location: str = "Town03_Roundabout"
    agent_count: int = 100
    batch_size: int = 1
    obs_length: int = 1
    steps: int = 1
    min_speed: int = 1  # Km/h
    max_speed: int = 5  # Km/h
    simulator: str = "None"
    get_infractions: bool = False


def initialize(
    location="CARLA:Town03:Roundabout",
    agent_count=1,
    batch_size=1,
    min_speed=1,
    max_speed=5,
) -> dict:
    start = time.time()
    timeout = TIMEOUT

    while True:
        try:
            initial_states = iai.session.initialize(
                location=location,
                agent_count=agent_count,
                batch_size=batch_size,
                min_speed=np.ceil(min_speed / 3.6).astype(int),
                max_speed=np.ceil(max_speed / 3.6).astype(int),
            )
            response = {
                "states": initial_states["initial_condition"]["agent_states"],
                "recurrent_states": None,
                "attributes": initial_states["initial_condition"]["agent_sizes"],
            }
            return response
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))


def drive(
    states: dict,
    agent_attributes: dict,
    recurrent_states: Optional[InputDataType] = None,
    get_birdviews: bool = False,
    location="CARLA:Town03:Roundabout",
    steps: int = 1,
    get_infractions: bool = False,
) -> dict:
    def _validate(input_dict: dict, input_name: str):
        input_data = input_dict[input_name]
        if isinstance(input_data, list):
            input_data = torch.Tensor(input_data)
        if input_data.shape[1] != agent_count:
            raise Exception(f"{input_name} has the wrong agent counts (dim 1)")
        if len(input_data.shape) > 2:
            # TODO: We hide the time dimension of the present masks for the client for now
            pass
        return input_data

    def _validate_recurrent_states(input_data: InputDataType):
        if isinstance(input_data, list):
            input_data = torch.Tensor(input_data)
        if input_data.shape[1] != agent_count:
            raise Exception("Recurrent states has the wrong agent counts (dim 2)")
        if input_data.shape[2] != 2:
            raise Exception("Recurrent states has the wrong number of layers (dim 4)")
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

    agent_count = len(states[0])
    recurrent_states = (
        _tolist(_validate_recurrent_states(recurrent_states))
        if recurrent_states is not None
        else None
    )  # Bx(num_predictions)xAxTx2x64

    model_inputs = dict(
        location=location,
        past_observations=dict(
            agent_states=states,
            agent_sizes=agent_attributes,
        ),
        recurrent_states=recurrent_states,
        # Expand from BxA to BxAxT_total for the API interface
        steps=steps,
        get_birdviews=get_birdviews,
        get_infractions=get_infractions,
    )

    start = time.time()
    timeout = TIMEOUT

    while True:
        try:
            return iai.session.run(model_inputs)
        except Exception as e:
            # TODO: Add logger
            iai.logger.warning("Retrying")
            if timeout is not None and time.time() > start + timeout:
                raise e
