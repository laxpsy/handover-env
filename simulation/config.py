from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # rendering parameters
    SCREEN_WIDTH: int = 800
    SCREEN_HEIGHT: int = 600
    STEP: int = 100
    # path loss model parameters
    PATH_LOSS_EXPONENT: float = 3
    PATH_LOSS_REFERENCE_DISTANCE: float = 1
    # calculation constants
    SPEED_OF_LIGHT: float = 299_792_458
    # handover parameters
    HYSTERISIS_MARGIN: float = 3
    TIME_TO_TRIGGER: float = 3
    PING_PONG_WINDOW: float = 3
    RLF_FAILURE_THRESHOLD: float = -100
    # ue history parameter
    MAX_HISTORY: int = 10
    # handover type detection parameters
    EARLY_HANDOVER_WINDOW: int = 1000
    MIN_HISTORY_LENGTH: int = 3
    # ue defaults
    DEFAULT_VELOCITY: float = 1.0
    # base station defaults
    DEFAULT_TX_POWER: float = 43.0
    DEFAULT_FREQUENCY: float = 2.1e9
