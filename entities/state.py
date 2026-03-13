from dataclasses import dataclass
from typing import List

from entities.base_station import BaseStation
from entities.ue import UE


@dataclass
class Network:
    ues: List[UE]
    base_stations: List[BaseStation]
