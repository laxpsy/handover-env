import logging

from entities.network import Network
from entities.simulation_events import SimulationEvents
from entities.ue import UE
from loggers.logger_helpers import log_event
from simulation.config import SimulationConfig

logger = logging.getLogger("HANDOVER_ENV")


def naive_handover(
    ue: UE, network: Network, config: SimulationConfig, step_count: int
) -> int | None:
    serving_bs = ue.serving_bs
    if serving_bs is None:
        return None

    serving_bs_rsrp = ue.rsrp[serving_bs]

    best_bs = serving_bs
    best_rsrp = serving_bs_rsrp

    for bs in network.base_stations:
        if ue.rsrp[bs.id] > best_rsrp:
            best_bs = bs.id
            best_rsrp = ue.rsrp[bs.id]

    rsrp_delta = best_rsrp - serving_bs_rsrp

    if ue.handover_state.ttt_running:
        if ue.handover_state.target_base_station == best_bs:
            if ue.handover_state.ttt_timer > 0:
                ue.handover_state.ttt_timer -= 1
            if ue.handover_state.ttt_timer <= 0:
                log_event(
                    step_count,
                    ue.id,
                    SimulationEvents.HANDOVER.value,
                    best_bs=best_bs,
                )
                return best_bs
        else:
            ue.handover_state.ttt_running = False
            ue.handover_state.ttt_timer = 0.0

    if rsrp_delta > config.HYSTERISIS_MARGIN:
        ue.handover_state.target_base_station = best_bs
        ue.handover_state.ttt_running = True
        ue.handover_state.ttt_timer = config.TIME_TO_TRIGGER

    return None
