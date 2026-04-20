"""
State encoding utilities for RL agents.
"""
import numpy as np
from entities.ue import UE
from entities.network import Network

def encode_state_discrete(ue: UE, network: Network) -> tuple:
    # ...existing Q-Learning discretization logic...
    serving_rsrp = ue.rsrp.get(ue.serving_bs, -120)
    candidate_rsrps = {
        bs_id: rsrp for bs_id, rsrp in ue.rsrp.items() if bs_id != ue.serving_bs
    }
    best_candidate_rsrp = max(candidate_rsrps.values()) if candidate_rsrps else -120
    delta = best_candidate_rsrp - serving_rsrp
    rsrp_bin = int(np.digitize(serving_rsrp, [-97, -72, -60]))
    rsrp_bin = min(rsrp_bin, 3)
    delta_bin = int(np.digitize(delta, [-5, 1, 5]))
    delta_bin = min(delta_bin, 3)
    ttt_bin = int(ue.handover_state.ttt_running)
    return (rsrp_bin, delta_bin, ttt_bin)

def encode_state_continuous(ue: UE, network: Network) -> np.ndarray:
    # ...existing REINFORCE continuous encoding logic...
    serving_rsrp = ue.rsrp.get(ue.serving_bs, -120.0)
    candidate_rsrps = sorted(
        [rsrp for bs_id, rsrp in ue.rsrp.items() if bs_id != ue.serving_bs],
        reverse=True
    )
    while len(candidate_rsrps) < 3:
        candidate_rsrps.append(-120.0)
    candidate_rsrps = candidate_rsrps[:3]
    best_candidate = candidate_rsrps[0]
    delta = best_candidate - serving_rsrp
    def norm_rsrp(r): return (r + 80.0) / 40.0
    return np.array([
        norm_rsrp(serving_rsrp),
        norm_rsrp(best_candidate),
        delta / 20.0,
        float(ue.handover_state.ttt_running),
        min(ue.handover_state.step_count_since_last_handover, 30) / 30.0
    ], dtype=np.float32)
