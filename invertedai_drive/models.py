from dataclasses import dataclass
from typing import List
from torch import Tensor


@dataclass
class AgentStatesWithSample:
    x: List[List[List[List[float]]]]  # (num_predictions)xBxAxT_obs
    y: List[List[List[List[float]]]]  # (num_predictions)xBxAxT_obs
    psi: List[List[List[List[float]]]]  # (num_predictions)xBxAxT_obs
    speed: List[List[List[List[float]]]]  # (num_predictions)xBxAxT_obs


@dataclass
class DriveResponse:
    states: AgentStatesWithSample
    recurrent_states: List[
        List[List[List[List[List[float]]]]]
    ]  # (num_predictions)xBxAxTx2x64
    bird_view: List[int]
