from dataclasses import dataclass
from typing import Callable

from entities.network import Network
from entities.ue import UE


@dataclass
class HandoverPolicy:
    decision_function: Callable[[UE, Network], int]

    def __call__(self, ue: UE, state_space: Network) -> int:
        return self.decision_function(ue, state_space)
