from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from algorithms.handover import naive_handover
from entities.base_station import BaseStation
from entities.handover_policy import HandoverPolicy
from entities.network import Network
from entities.rendering_entities import Coordinates
from entities.ue import UE, UEMovementType
from entities.handover import UEHandoverState
from simulation.config import SimulationConfig
from simulation.simulator import Simulation
from simulation.statistics import SimulationStatistics


DEFAULT_CONFIG = SimulationConfig(
    SCREEN_WIDTH=800,
    SCREEN_HEIGHT=600,
    STEP=100,
    HYSTERISIS_MARGIN=3,
    TIME_TO_TRIGGER=3,
    PING_PONG_WINDOW=3,
    RLF_FAILURE_THRESHOLD=-100,
    MAX_HISTORY=10,
    EARLY_HANDOVER_WINDOW=1000,
    MIN_HISTORY_LENGTH=3,
    DEFAULT_VELOCITY=1.0,
    DEFAULT_TX_POWER=43.0,
    DEFAULT_FREQUENCY=2.1e9,
)

HIGH_MOBILITY_CONFIG = SimulationConfig(
    SCREEN_WIDTH=800,
    SCREEN_HEIGHT=600,
    STEP=100,
    HYSTERISIS_MARGIN=4,
    TIME_TO_TRIGGER=5,
    PING_PONG_WINDOW=4,
    RLF_FAILURE_THRESHOLD=-100,
    MAX_HISTORY=10,
    EARLY_HANDOVER_WINDOW=1000,
    MIN_HISTORY_LENGTH=3,
    DEFAULT_VELOCITY=3.0,
    DEFAULT_TX_POWER=43.0,
    DEFAULT_FREQUENCY=2.1e9,
)

LOW_LATENCY_CONFIG = SimulationConfig(
    SCREEN_WIDTH=800,
    SCREEN_HEIGHT=600,
    STEP=100,
    HYSTERISIS_MARGIN=2,
    TIME_TO_TRIGGER=1,
    PING_PONG_WINDOW=2,
    RLF_FAILURE_THRESHOLD=-100,
    MAX_HISTORY=10,
    EARLY_HANDOVER_WINDOW=500,
    MIN_HISTORY_LENGTH=3,
    DEFAULT_VELOCITY=1.5,
    DEFAULT_TX_POWER=43.0,
    DEFAULT_FREQUENCY=2.1e9,
)


class HandoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_ues: int = 1,
        num_bs: int = 3,
        config: SimulationConfig = DEFAULT_CONFIG,
    ):
        super().__init__()
        self.num_ues = num_ues
        self.num_bs = num_bs
        self.config = config
        self.handover_policy = HandoverPolicy(naive_handover)

        self.observation_space = spaces.Dict(
            {
                "ue_positions": spaces.Box(
                    low=0,
                    high=max(self.config.SCREEN_WIDTH, self.config.SCREEN_HEIGHT),
                    shape=(num_ues, 2),
                    dtype=np.float32,
                ),
                "ue_velocities": spaces.Box(
                    low=-10,
                    high=10,
                    shape=(num_ues, 2),
                    dtype=np.float32,
                ),
                "rsrp": spaces.Box(
                    low=-200,
                    high=0,
                    shape=(num_ues, num_bs),
                    dtype=np.float32,
                ),
                "serving_bs": spaces.Box(
                    low=0,
                    high=num_bs,
                    shape=(num_ues,),
                    dtype=np.int32,
                ),
                "time_since_handover": spaces.Box(
                    low=0,
                    high=10000,
                    shape=(num_ues,),
                    dtype=np.int32,
                ),
            }
        )

        self.action_space = spaces.Discrete(num_bs + 1)

        self.network = self._create_network()
        self.statistics = SimulationStatistics()
        self.simulation = Simulation(
            config=self.config,
            statistics=self.statistics,
            handover_policy=self.handover_policy,
            state_space=self.network,
        )
        self._current_step = 0

    def _create_network(self) -> Network:
        ues = []
        for i in range(self.num_ues):
            ue = UE(
                id=i,
                coordinates=Coordinates(
                    x=np.random.rand() * self.config.SCREEN_WIDTH,
                    y=np.random.rand() * self.config.SCREEN_HEIGHT,
                ),
                velocity_x=self.config.DEFAULT_VELOCITY,
                velocity_y=self.config.DEFAULT_VELOCITY,
                serving_bs=0,
                rsrp={},
                movement_type=UEMovementType.Linear,
                handover_state=UEHandoverState(
                    target_base_station=-1,
                    ttt_timer=0.0,
                    ttt_running=False,
                    step_count_since_last_handover=0,
                    handover_this_step=False,
                ),
                handover_history=[],
                total_handovers=0,
            )
            ues.append(ue)

        base_stations = []
        for i in range(self.num_bs):
            bs = BaseStation(
                id=i,
                coordinates=Coordinates(
                    x=np.random.rand() * self.config.SCREEN_WIDTH,
                    y=np.random.rand() * self.config.SCREEN_HEIGHT,
                ),
                tx_power=self.config.DEFAULT_TX_POWER,
                transmission_frequency=self.config.DEFAULT_FREQUENCY,
            )
            base_stations.append(bs)

        return Network(ues=ues, base_stations=base_stations)

    def _get_observation(self) -> dict[str, Any]:
        positions = np.array(
            [[ue.coordinates.x, ue.coordinates.y] for ue in self.network.ues],
            dtype=np.float32,
        )
        velocities = np.array(
            [[ue.velocity_x, ue.velocity_y] for ue in self.network.ues],
            dtype=np.float32,
        )
        rsrp = np.array(
            [
                [ue.rsrp.get(bs.id, -150) for bs in self.network.base_stations]
                for ue in self.network.ues
            ],
            dtype=np.float32,
        )
        serving_bs = np.array(
            [
                ue.serving_bs if ue.serving_bs is not None else -1
                for ue in self.network.ues
            ],
            dtype=np.int32,
        )
        time_since = np.array(
            [
                ue.handover_state.step_count_since_last_handover
                for ue in self.network.ues
            ],
            dtype=np.int32,
        )

        return {
            "ue_positions": positions,
            "ue_velocities": velocities,
            "rsrp": rsrp,
            "serving_bs": serving_bs,
            "time_since_handover": time_since,
        }

    def _compute_reward(self) -> float:
        reward = 0.0
        reward -= self.statistics.ping_pong_handover_count * 0.5
        reward -= self.statistics.early_handover_count * 0.3
        reward -= self.statistics.late_handover_count * 0.3
        return reward

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Any], dict]:
        super().reset(seed=seed)
        self._current_step = 0
        self.network = self._create_network()
        self.statistics = SimulationStatistics()
        self.simulation = Simulation(
            config=self.config,
            statistics=self.statistics,
            handover_policy=self.handover_policy,
            state_space=self.network,
        )
        return self._get_observation(), {}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict]:
        self._current_step += 1

        if action > 0 and action <= self.num_bs:
            target_bs_id = action - 1
            ue = self.network.ues[0]
            if ue.serving_bs != target_bs_id:
                from core.handover_helpers import perform_handover

                target_bs = self.network.base_stations[target_bs_id]
                perform_handover(
                    ue, target_bs, self.config, self.simulation.get_step_count()
                )

        self.simulation.step()

        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = False
        truncated = False
        info = {
            "step_count": self._current_step,
            "statistics": {
                "total_handovers": self.simulation.state_space.ues[0].total_handovers,
                "ping_pong": self.statistics.ping_pong_handover_count,
                "early": self.statistics.early_handover_count,
                "late": self.statistics.late_handover_count,
            },
            "config": self.simulation.get_config(),
        }

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        pass
