from dataclasses import dataclass


@dataclass
class UEHandoverState:
    target_base_station: int
    ttt_timer: float
    ttt_running: bool
    step_count_since_last_handover: int
    handover_this_step: bool = False
