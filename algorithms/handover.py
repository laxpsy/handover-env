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

    if not network.base_stations:
        return None

    best_bs = network.base_stations[0].id
    best_rsrp = ue.rsrp.get(best_bs, float("-inf"))

    for bs in network.base_stations:
        bs_rsrp = ue.rsrp.get(bs.id, float("-inf"))
        if bs_rsrp > best_rsrp:
            best_bs = bs.id
            best_rsrp = bs_rsrp

    if serving_bs is None:
        if best_rsrp < config.RLF_FAILURE_THRESHOLD:
            ue.handover_state.ttt_running = False
            ue.handover_state.ttt_timer = 0.0
            ue.handover_state.target_base_station = -1
            return None

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

        ue.handover_state.target_base_station = best_bs
        ue.handover_state.ttt_running = True
        ue.handover_state.ttt_timer = config.TIME_TO_TRIGGER
        return None

    serving_bs_rsrp = ue.rsrp.get(serving_bs)
    if serving_bs_rsrp is None:
        return None

    if best_bs == serving_bs:
        best_rsrp = serving_bs_rsrp

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
