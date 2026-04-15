import logging
from typing import Optional

from entities.base_station import BaseStation
from entities.simulation_events import HandoverEventTypes, SimulationEvents
from entities.ue import UE
from loggers.logger_helpers import log_error, log_event
from simulation.config import SimulationConfig
from simulation.statistics import SimulationStatistics

logger = logging.getLogger("HANDOVER_ENV")


def check_handover_type(
    ue: UE,
    config: SimulationConfig,
    step_count: int,
    handover_happened: bool,
    statistics: SimulationStatistics,
):
    if ue.serving_bs is None:
        ue.handover_state.was_late_since_last_handover = False
        return

    ue_serving_bs_rsrp: Optional[float] = ue.rsrp.get(ue.serving_bs)

    if ue_serving_bs_rsrp is not None:
        if handover_happened:
            ue.handover_state.was_late_since_last_handover = False
            if (
                ue_serving_bs_rsrp < config.RLF_FAILURE_THRESHOLD
                and ue.handover_state.step_count_since_last_handover
                < config.EARLY_HANDOVER_WINDOW
            ):
                statistics.early_handover_count += 1
                log_event(
                    step_count,
                    ue.id,
                    SimulationEvents.HANDOVER_TYPE_DETECTION.value,
                    handover_type=HandoverEventTypes.HANDOVER_TOO_EARLY.value,
                )
            elif (
                len(ue.handover_history) >= config.MIN_HISTORY_LENGTH
                and ue.handover_history[-1] == ue.handover_history[-3]
                and ue.handover_state.step_count_since_last_handover
                <= config.PING_PONG_WINDOW
            ):
                statistics.ping_pong_handover_count += 1
                log_event(
                    step_count,
                    ue.id,
                    SimulationEvents.HANDOVER_TYPE_DETECTION.value,
                    handover_type=HandoverEventTypes.HANDOVER_PING_PONG.value,
                )
            else:
                statistics.successful_handover_count += 1
                log_event(
                    step_count,
                    ue.id,
                    SimulationEvents.HANDOVER_TYPE_DETECTION.value,
                    handover_type=HandoverEventTypes.HANDOVER_SUCCESS.value,
                )
        else:
            if ue_serving_bs_rsrp < config.RLF_FAILURE_THRESHOLD:
                if not ue.handover_state.was_late_since_last_handover:
                    statistics.late_handover_count += 1
                    log_event(
                        step_count,
                        ue.id,
                        SimulationEvents.HANDOVER_TYPE_DETECTION.value,
                        handover_type=HandoverEventTypes.HANDOVER_TOO_LATE.value,
                    )
                    ue.handover_state.was_late_since_last_handover = True
            else:
                ue.handover_state.was_late_since_last_handover = False


def perform_handover(
    ue: UE,
    target_bs: Optional[BaseStation],
    config: SimulationConfig,
    step_count: int,
):
    if target_bs is None:
        log_error(
            step_count,
            SimulationEvents.HANDOVER.value,
            ue=ue,
            reason="target_bs_is_none",
        )
        return

    ue.serving_bs = target_bs.id
    if len(ue.handover_history) == config.MAX_HISTORY:
        _ = ue.handover_history.pop(0)
    ue.handover_history.append(target_bs.id)
    ue.handover_state.step_count_since_last_handover = 0
    ue.handover_state.target_base_station = target_bs.id
    ue.handover_state.ttt_running = False
    ue.handover_state.ttt_timer = 0.0
    ue.handover_state.handover_this_step = True
    ue.handover_state.was_late_since_last_handover = False
    ue.handover_state.handover_flash_steps = 12
    ue.total_handovers += 1
    log_event(
        step_count,
        ue.id,
        SimulationEvents.HANDOVER.value,
        ue=ue,
        target_bs=target_bs,
    )
