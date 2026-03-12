from dataclasses import dataclass
from entities.rendering_entities import Coordinates
from enum import Enum

class BSType(Enum):
    LTE = "4G"
    NR = "5G"

@dataclass
class BaseStation:
    id: int
    coordinates: Coordinates
    tx_power: float
    transmission_frequency: float
    type: BSType
