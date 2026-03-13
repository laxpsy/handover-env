from dataclasses import dataclass


@dataclass
class SimulatorConfig:
    # rendering parameters
    SCREEN_WIDTH: int = 800
    SCREEN_HEIGHT: int = 600
    # path loss model parameters
    PATH_LOSS_EXPONENT: float = 3
    PATH_LOSS_REFERENCE_DISTANCE: float = 1
    # calculation constants
    SPEED_OF_LIGHT: float = 299_792_458
    # handover parameters
    HYSTERISIS_MARGIN: float = 3  # TODO
    TIME_TO_TRIGGER: float = 3  # TODO
    PING_PONG_WINDOW: float = 5000  # TODO ms
    RLF_FAILURE_THRESHOLD: float = -100  # TODO dBm
    # ue history parameter
    MAX_HISTORY: int = 10
