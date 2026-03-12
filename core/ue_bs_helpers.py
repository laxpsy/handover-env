from entities.rendering_entities import Coordinates
import numpy as np
from entities.base_station import BaseStation
from entities.ue import UE
from config import SimulatorConfig

def calculate_distance(c1: Coordinates, c2: Coordinates) -> float:
    dx = c2.x - c1.x
    dy = c2.y - c1.y

    return np.sqrt(dx*dx + dy*dy)

def calculate_rsrp(ue: UE, bs: BaseStation):
    path_loss_reference = 20*np.log10(
            4*np.pi*SimulatorConfig.PATH_LOSS_REFERENCE_DISTANCE/bs.transmission_frequency
            )  
    path_loss = path_loss_reference + 10 * SimulatorConfig.PATH_LOSS_EXPONENT * np.log10(calculate_distance(ue.coordinates, bs.coordinates)/SimulatorConfig.PATH_LOSS_REFERENCE_DISTANCE)
    return (bs.tx_power - path_loss)


