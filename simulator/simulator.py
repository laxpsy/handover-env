from dataclasses import dataclass

from entities.handover_policy import HandoverPolicy
from entities.network import Network
from simulator.config import SimulationConfig


@dataclass
class Simulation:
    config: SimulationConfig
    handover_policy: HandoverPolicy
    state_space: Network
    time: float = 0.0
    step_count: int = 0
