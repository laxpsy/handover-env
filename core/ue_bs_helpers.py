from entities.rendering_entities import Coordinates
import numpy as np
from entities.base_station import BaseStation
from entities.state import Network
from entities.ue import UE, UEMovementType
from config import SimulatorConfig


def calculate_distance(c1: Coordinates, c2: Coordinates) -> float:
    dx = c2.x - c1.x
    dy = c2.y - c1.y

    return max(SimulatorConfig.PATH_LOSS_REFERENCE_DISTANCE, np.sqrt(dx*dx + dy*dy))


def calculate_rsrp_ue_bs_pair(ue: UE, bs: BaseStation) -> float:
    # note: bs.tx_power should be in dBm
    transmission_wavelength = SimulatorConfig.SPEED_OF_LIGHT/bs.transmission_frequency
    if transmission_wavelength == 0:
        raise ValueError("transmission_wavelength is 0")
    path_loss_reference = 20*np.log10(
        4*np.pi*SimulatorConfig.PATH_LOSS_REFERENCE_DISTANCE/transmission_wavelength
    )
    path_loss = path_loss_reference + 10 * SimulatorConfig.PATH_LOSS_EXPONENT * \
        np.log10(calculate_distance(ue.coordinates, bs.coordinates) /
                 SimulatorConfig.PATH_LOSS_REFERENCE_DISTANCE)
    return (bs.tx_power - path_loss)


def mobility_update(state_space: Network) -> None:
    for ue in state_space.ues:
        if ue.movement_type == UEMovementType.Linear:
            ue.coordinates.x += ue.velocity_x
            ue.coordinates.y += ue.velocity_y
            if ue.coordinates.x <= 0 or ue.coordinates.x >= SimulatorConfig.SCREEN_WIDTH:
                ue.velocity_x *= -1
            if ue.coordinates.y <= 0 or ue.coordinates.y >= SimulatorConfig.SCREEN_HEIGHT:
                ue.velocity_y *= -1
        elif ue.movement_type == UEMovementType.Random:
            # TODO
            continue


def calculate_rsrp_naive(state_space: Network) -> None:
    for ue in state_space.ues:
        ue.rsrp.clear()
        for bs in state_space.base_stations:
            ue.rsrp[bs.id] = calculate_rsrp_ue_bs_pair(ue, bs)
