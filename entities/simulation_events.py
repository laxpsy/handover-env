from enum import Enum


class SimulationEvents(Enum):
    # ties in to perform_handover()
    HANDOVER: "HANDOVER"
    # ties in to detect_handover_type()
    HANDOVER_TYPE_DETECTION: "HANDOVER_TYPE_DETECTION"
