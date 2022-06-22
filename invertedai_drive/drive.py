from invertedai_drive.utils import Client
import torch
import numpy as np
from typing import List, Union, Optional

InputDataType = Union[torch.Tensor, np.ndarray, List]
client = Client()


def run(
        api_key: str,
        model_key: str,
        location: str,
        x: InputDataType,
        y: InputDataType,
        psi: InputDataType,
        speed: InputDataType,
        length: InputDataType,
        width: InputDataType,
        lr: InputDataType,
        recurrent_states: Optional[InputDataType],
        present_masks: Optional[InputDataType],
        batch_size: int,
        agent_counts: int,
        obs_length: int,
        step_times: int = 1,
        num_predictions: int = 1,
        ):

    def _validate(input_data: InputDataType, input_name: str):
        if isinstance(input_data, list):
            input_data = torch.Tensor(input_data)
        if input_data.shape[0] != batch_size:
            raise Exception(f"{input_name} has the wrong batch size (dim 0). "
                            f"Expect {batch_size}, but got {input_data.shape[0]}")
        if input_data.shape[1] != agent_counts:
            raise Exception(f"{input_name} has the wrong agent counts (dim 1). "
                            f"Expect {agent_counts}, but got {input_data.shape[1]}")
        if len(input_data.shape) > 2:
            if input_data.shape[2] != obs_length:
                raise Exception(f"{input_name} has the wrong time observation length (dim 2)"
                                f"Expect {obs_length}, but got {input_data.shape[2]}")
        return input_data

    def _validate_recurrent_states(input_data: InputDataType):
        if isinstance(input_data, list):
            input_data = torch.Tensor(input_data)
        if input_data.shape[0] != batch_size:
            raise Exception(f"Recurrent states has the wrong batch size (dim 0). "
                            f"Expect {batch_size}, but got {input_data.shape[0]}")
        if input_data.shape[1] != agent_counts:
            raise Exception(f"Recurrent states has the wrong agent counts (dim 1). "
                            f"Expect {agent_counts}, but got {input_data.shape[1]}")
        if input_data.shape[2] != 2:
            raise Exception(f"Recurrent states has the wrong number of layers (dim 2). "
                            f"Expect 2, but got {input_data.shape[2]}")
        if input_data.shape[3] != 64:
            raise Exception(f"Recurrent states has the wrong dimension (dim 3). "
                            f"Expect 64, but got {input_data.shape[3]}")
        return input_data

    def _tolist(input_data: InputDataType):
        if not isinstance(input_data, list):
            return input_data.tolist()
        else:
            return input_data

    def _validate_and_tolist(input_data: InputDataType, input_name: str):
        return _tolist(_validate(input_data, input_name))

    x = _validate_and_tolist(x, "x")  # BxAxT
    y = _validate_and_tolist(y, "y")  # BxAxT
    psi = _validate_and_tolist(psi, "psi")  # BxAxT
    speed = _validate_and_tolist(speed, "speed")  # BxAxT
    agent_length = _validate_and_tolist(length, "agent_length")  # BxA
    agent_width = _validate_and_tolist(width, "agent_width")  # BxA
    agent_lr = _validate_and_tolist(lr, "agent_lr")  # BxA
    present_masks = (
        _validate_and_tolist(present_masks, "present_masks")
        if present_masks is not None
        else None
    )  # BxAxT_obs
    recurrent_states = (
        _tolist(_validate_recurrent_states(recurrent_states))
        if recurrent_states is not None
        else None
    )  # BxAx2x64

    model_inputs = dict(
        location=location,
        initial_conditions=dict(
            agent_states=dict(x=x, y=y, psi=psi, speed=speed),
            agent_sizes=dict(length=agent_length, width=agent_width, lr=agent_lr),
        ),
        recurrent_states=recurrent_states,
        present_masks=present_masks,
        batch_size=batch_size,
        agent_counts=agent_counts,
        obs_length=obs_length,
        step_times=step_times,
        num_predictions=num_predictions,
    )

    output = client.run(api_key, model_key, model_inputs)

    return output
