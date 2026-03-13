from dataclasses import dataclass
from entities.handover import UEHandoverState
from entities.rendering_entities import Coordinates
from enum import Enum


class UEMovementType(Enum):
    Linear = 0
    Random = 1


@dataclass
class UE:
    id: int
    coordinates: Coordinates
    velocity_x: float
    velocity_y: float
    serving_bs: int | None
    rsrp: dict[int, float]
    movement_type: UEMovementType
    handover_state: UEHandoverState
    handover_history: list[int]
    total_handovers: int
