from dataclasses import dataclass
from typing import Optional, Dict

from entities.rendering_entities import Coordinates

@dataclass
class UE:
    id: int
    coordinates: Coordinates
    velocity_x: float
    velocity_y: float
    serving_bs: Optional[int]
    rsrp: Dict[int, int]

