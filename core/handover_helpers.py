import logging
from logging.logger_helpers import log_error, log_event
from typing import Optional

from sim.config import SimulationConfig

from entities.base_station import BaseStation
from entities.handover import HandoverEventTypes
from entities.simulation_events import SimulationEvents
from entities.ue import UE

logger = logging.getLogger("HANDOVER_ENV")


def detect_handover_type(ue: UE, handover_happened: bool):
    if ue.serving_bs is None:
        # TODO: add step
        log_error(0, SimulationEvents.HANDOVER_TYPE_DETECTION, ue, handover_happened)
        return HandoverEventTypes.HANDOVER_NONE

    # should never be None, if serving_bs is non-null, RSRP will exist
    ue_serving_bs_rsrp: Optional[float] = ue.rsrp.get(ue.serving_bs)

    if ue_serving_bs_rsrp is not None:
        if handover_happened:
            # case-1, too early
            if (
                ue_serving_bs_rsrp < SimulationConfig.RLF_FAILURE_THRESHOLD
                and ue.handover_state.time_since_last_handover < 1000.0
            ):
                return HandoverEventTypes.HANDOVER_TOO_EARLY
            elif (
                len(ue.handover_history) >= 3
                and ue.handover_history[-1] == ue.handover_history[-3]
                and ue.handover_state.time_since_last_handover
                <= SimulationConfig.PING_PONG_WINDOW
            ):
                return HandoverEventTypes.HANDOVER_PING_PONG

            else:
                return HandoverEventTypes.HANDOVER_SUCCESS
        else:
            if ue_serving_bs_rsrp < SimulationConfig.RLF_FAILURE_THRESHOLD:
                return HandoverEventTypes.HANDOVER_TOO_LATE

    # TODO: add step
    log_event(0, ue.id, SimulationEvents.HANDOVER_TYPE_DETECTION, ue, handover_happened)
    # base-case, should never happen ideally
    return HandoverEventTypes.HANDOVER_NONE


def perform_handover(ue: UE, target_bs: BaseStation):
    ue.serving_bs = target_bs.id
    if len(ue.handover_history) == SimulationConfig.MAX_HISTORY:
        _ = ue.handover_history.pop(0)
    ue.handover_history.append(target_bs.id)
    ue.handover_state.time_since_last_handover = 0.0
    ue.handover_state.target_base_station = target_bs.id
    ue.handover_state.ttt_running = True
    ue.handover_state.ttt_timer = 0.0
    ue.total_handovers += 1
    # TODO: add step
    log_event(0, ue.id, SimulationEvents.HANDOVER, ue, target_bs)
