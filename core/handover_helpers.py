from typing import Optional
from config import SimulatorConfig
from entities.handover import HandoverEventTypes
from entities.ue import UE
from entities.base_station import BaseStation


def detect_handover_type(ue: UE, handover_happened: bool):
    if ue.serving_bs is None:
        # ERROR
        return HandoverEventTypes.HANDOVER_NONE

    # should never be None, if serving_bs is non-null, RSRP will exist
    ue_serving_bs_rsrp: Optional[float] = ue.rsrp.get(ue.serving_bs)

    if ue_serving_bs_rsrp is not None:
        if handover_happened:
            # case-1, too early
            if ue_serving_bs_rsrp < SimulatorConfig.RLF_FAILURE_THRESHOLD and ue.handover_state.time_since_last_handover < 1000.0:
                return HandoverEventTypes.HANDOVER_TOO_EARLY
            elif len(ue.handover_history) >= 3 and ue.handover_history[-1] == ue.handover_history[-3] and ue.handover_state.time_since_last_handover <= SimulatorConfig.PING_PONG_WINDOW:
                return HandoverEventTypes.HANDOVER_PING_PONG
            else:
                return HandoverEventTypes.HANDOVER_SUCCESS
        else:
            if ue_serving_bs_rsrp < SimulatorConfig.RLF_FAILURE_THRESHOLD:
                return HandoverEventTypes.HANDOVER_TOO_LATE


def perform_handover(ue: UE, target_bs: BaseStation):
    ue.serving_bs = target_bs.id
    if len(ue.handover_history) == SimulatorConfig.MAX_HISTORY:
        _ = ue.handover_history.pop(0)
    ue.handover_history.append(target_bs.id)
    ue.handover_state.time_since_last_handover = 0.0
    ue.handover_state.target_base_station = target_bs.id
    ue.handover_state.ttt_running = True
    ue.handover_state.ttt_timer = 0.0
    ue.total_handovers += 1
