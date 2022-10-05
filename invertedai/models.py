from dataclasses import dataclass
from typing import List


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
    collision = List[List[List[float]]]  # T_obsxBxA
    offroad = List[List[List[float]]]  # T_obsxBxA
    wrong_way = List[List[List[float]]]  # T_obsxBxA
