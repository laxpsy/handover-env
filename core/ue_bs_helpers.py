import numpy as np
from simulation.config import SimulationConfig

from core.handover_helpers import perform_handover
from entities.base_station import BaseStation
from entities.handover_policy import HandoverPolicy
from entities.network import Network
from entities.rendering_entities import Coordinates
from entities.ue import UE, UEMovementType


def calculate_distance(c1: Coordinates, c2: Coordinates) -> float:
    dx = c2.x - c1.x
    dy = c2.y - c1.y

    return max(
        SimulationConfig.PATH_LOSS_REFERENCE_DISTANCE,
        np.sqrt(dx * dx + dy * dy),
    )


def calculate_rsrp_ue_bs_pair(
    ue: UE, bs: BaseStation, config: SimulationConfig
) -> float:
    transmission_wavelength = config.SPEED_OF_LIGHT / bs.transmission_frequency
    if transmission_wavelength == 0:
        raise ValueError("transmission_wavelength is 0")
    path_loss_reference = 20 * np.log10(
        4 * np.pi * config.PATH_LOSS_REFERENCE_DISTANCE / transmission_wavelength
    )
    path_loss = path_loss_reference + 10 * config.PATH_LOSS_EXPONENT * np.log10(
        calculate_distance(ue.coordinates, bs.coordinates)
        / config.PATH_LOSS_REFERENCE_DISTANCE
    )
    return bs.tx_power - path_loss


def mobility_update(state_space: Network, config: SimulationConfig) -> None:
    for ue in state_space.ues:
        if ue.movement_type == UEMovementType.Linear:
            ue.coordinates.x += ue.velocity_x
            ue.coordinates.y += ue.velocity_y
            if ue.coordinates.x <= 0 or ue.coordinates.x >= config.SCREEN_WIDTH:
                ue.velocity_x *= -1
            if ue.coordinates.y <= 0 or ue.coordinates.y >= config.SCREEN_HEIGHT:
                ue.velocity_y *= -1
        elif ue.movement_type == UEMovementType.Random:
            angle = np.arctan2(ue.velocity_y, ue.velocity_x)
            angle += np.random.uniform(-np.pi / 4, np.pi / 4)
            speed = np.sqrt(ue.velocity_x**2 + ue.velocity_y**2)
            speed = max(0.5, min(speed, 5.0))
            ue.velocity_x = speed * np.cos(angle)
            ue.velocity_y = speed * np.sin(angle)
            ue.coordinates.x += ue.velocity_x
            ue.coordinates.y += ue.velocity_y
            if ue.coordinates.x <= 0:
                ue.coordinates.x = 0
                ue.velocity_x *= -1
            elif ue.coordinates.x >= config.SCREEN_WIDTH:
                ue.coordinates.x = config.SCREEN_WIDTH
                ue.velocity_x *= -1
            if ue.coordinates.y <= 0:
                ue.coordinates.y = 0
                ue.velocity_y *= -1
            elif ue.coordinates.y >= config.SCREEN_HEIGHT:
                ue.coordinates.y = config.SCREEN_HEIGHT
                ue.velocity_y *= -1


def update_timers(state_space: Network) -> None:
    for ue in state_space.ues:
        ue.handover_state.step_count_since_last_handover += 1


def calculate_rsrp_naive(state_space: Network, config: SimulationConfig) -> None:
    for ue in state_space.ues:
        ue.rsrp.clear()
        for bs in state_space.base_stations:
            ue.rsrp[bs.id] = calculate_rsrp_ue_bs_pair(ue, bs, config)


def decide_handovers(
    state_space: Network,
    config: SimulationConfig,
    policy: HandoverPolicy,
    step_count: int,
) -> None:
    for ue in state_space.ues:
        target_bs_id = policy(ue, state_space, config, step_count)
        if target_bs_id is not None:
            target_bs = state_space.base_stations[target_bs_id]
            perform_handover(ue, target_bs, config, step_count)
