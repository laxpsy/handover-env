from dataclasses import dataclass
from entities.rendering_entities import Coordinates


@dataclass
class BaseStation:
    id: int
    coordinates: Coordinates
    tx_power: float
    transmission_frequency: float
