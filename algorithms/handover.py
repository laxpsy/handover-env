from entities.network import Network
from entities.ue import UE

# TOOD: implement TTT handling


def naive_handover(ue: UE, network: Network):
    serving_bs = ue.serving_bs
    serving_bs_rsrp = ue.rsrp[serving_bs]

    best_bs = serving_bs
    best_rsrp = serving_bs_rsrp

    for bs in network.base_stations:

        if ue.rsrp[bs.id] > best_rsrp:
            best_bs = bs
            best_rsrp = ue.rsrp[bs.id]
