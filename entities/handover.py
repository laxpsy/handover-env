from dataclasses import dataclass


@dataclass
class UEHandoverState:
    target_base_station: int
    ttt_timer: float
    ttt_running: bool
    step_count_since_last_handover: int
    handover_this_step: bool = False
    was_late_since_last_handover: bool = False
    handover_flash_steps: int = 0
