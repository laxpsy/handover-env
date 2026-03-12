from dataclasses import dataclass

@dataclass
class SimulatorConfig:
    SCREEN_WIDTH: int = 800
    SCREEN_HEIGHT: int = 600
    PATH_LOSS_EXPONENT: float = 3
    PATH_LOSS_REFERENCE_DISTANCE: float = 1
