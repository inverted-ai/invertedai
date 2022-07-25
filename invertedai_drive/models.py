from dataclasses import dataclass
from typing import List
from torch import Tensor


@dataclass
class AgentStatesWithSample:
    x: List[List[List[float]]]  # BxAxT_obs
    y: List[List[List[float]]]  # BxAxT_obs
    psi: List[List[List[float]]]  # BxAxT_obs
    speed: List[List[List[float]]]  # BxAxT_obs


@dataclass
class DriveResponse:
    states: AgentStatesWithSample
    recurrent_states: List[List[List[List[List[float]]]]]  # BxAx2x64
    bird_view: List[int]
