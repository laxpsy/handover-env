from dataclasses import dataclass


@dataclass
class SimulationStatistics:
    early_handover_count: int = 0
    late_handover_count: int = 0
    ping_pong_handover_count: int = 0
    successful_handover_count: int = 0
