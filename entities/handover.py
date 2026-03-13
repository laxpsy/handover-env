from dataclasses import dataclass
from enum import Enum


class HandoverEventTypes(Enum):
    HANDOVER_SUCCESS = 0
    HANDOVER_TOO_EARLY = 1
    HANDOVER_TOO_LATE = 2
    HANDOVER_PING_PONG = 3
    HANDOVER_NONE = -1


@dataclass
class UEHandoverState:
    target_base_station: int
    ttt_timer: float
    ttt_running: bool
    time_since_last_handover: float
