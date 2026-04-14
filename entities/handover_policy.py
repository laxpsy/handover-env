from dataclasses import dataclass
from typing import Callable

from entities.network import Network
from entities.ue import UE
from simulation.config import SimulationConfig


@dataclass
class HandoverPolicy:
    decision_function: Callable[[UE, Network, SimulationConfig, int], int | None]

    def __call__(
        self, ue: UE, state_space: Network, config: SimulationConfig, step_count: int
    ) -> int | None:
        return self.decision_function(ue, state_space, config, step_count)
