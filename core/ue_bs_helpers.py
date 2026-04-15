import numpy as np

from core.handover_helpers import perform_handover
from entities.base_station import BaseStation
from entities.handover_policy import HandoverPolicy
from entities.network import Network
from entities.rendering_entities import Coordinates
from entities.simulation_events import SimulationEvents
from entities.ue import UE, UEMovementType
from loggers.logger_helpers import log_event
from simulation.config import SimulationConfig
from simulation.statistics import SimulationStatistics


def calculate_distance(
    c1: Coordinates, c2: Coordinates, config: SimulationConfig
) -> float:
    dx = c2.x - c1.x
    dy = c2.y - c1.y

    return max(
        config.PATH_LOSS_REFERENCE_DISTANCE,
        np.sqrt(dx * dx + dy * dy),
    )


def calculate_rsrp_ue_bs_pair(
    ue: UE, bs: BaseStation, config: SimulationConfig
) -> float:
    if bs.transmission_frequency == 0:
        raise ValueError("transmission_frequency is 0")
    transmission_wavelength = config.SPEED_OF_LIGHT / bs.transmission_frequency
    path_loss_reference = 20 * np.log10(
        4 * np.pi * config.PATH_LOSS_REFERENCE_DISTANCE / transmission_wavelength
    )
    path_loss = path_loss_reference + 10 * config.PATH_LOSS_EXPONENT * np.log10(
        calculate_distance(ue.coordinates, bs.coordinates, config)
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
        if ue.handover_state.handover_flash_steps > 0:
            ue.handover_state.handover_flash_steps -= 1


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
    statistics: SimulationStatistics,
) -> None:
    for ue in state_space.ues:
        if not state_space.base_stations:
            ue.serving_bs = None
            ue.handover_state.ttt_running = False
            ue.handover_state.ttt_timer = 0.0
            ue.handover_state.target_base_station = -1
            continue

        best_bs_id = max(ue.rsrp, key=ue.rsrp.get)
        best_rsrp = ue.rsrp[best_bs_id]

        if best_rsrp < config.RLF_FAILURE_THRESHOLD:
            if ue.serving_bs is not None:
                ue.serving_bs = None
                ue.handover_state.ttt_running = False
                ue.handover_state.ttt_timer = 0.0
                ue.handover_state.target_base_station = -1
                ue.handover_state.was_late_since_last_handover = False
                statistics.late_handover_count += 1
                log_event(
                    step_count,
                    ue.id,
                    SimulationEvents.HANDOVER_TYPE_DETECTION.value,
                    handover_type="LINK_TERMINATED",
                    best_rsrp=f"{best_rsrp:.2f}",
                )
            continue

        target_bs_id = policy(ue, state_space, config, step_count)
        if target_bs_id is not None:
            target_bs = next(
                (bs for bs in state_space.base_stations if bs.id == target_bs_id),
                None,
            )
            perform_handover(ue, target_bs, config, step_count)
