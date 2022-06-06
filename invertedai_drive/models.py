from dataclasses import dataclass
from torch import Tensor

@dataclass
class ModelOutputs:
    states: Tensor  # BxGxAxTx4
    recurrent_states: Tensor  # BxGxAxTxNxD
