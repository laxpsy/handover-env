import logging
from dataclasses import dataclass
from loggers.logger_helpers import log_error

from core.handover_helpers import check_handover_type
from core.son import SON
from core.ue_bs_helpers import (
    calculate_rsrp_naive,
    decide_handovers,
    mobility_update,
    update_timers,
)
from entities.handover_policy import HandoverPolicy
from entities.network import Network
from simulation.config import SimulationConfig
from simulation.statistics import SimulationStatistics

logger = logging.getLogger("HANDOVER_ENV")


@dataclass
class Simulation:
    config: SimulationConfig
    statistics: SimulationStatistics
    handover_policy: HandoverPolicy
    state_space: Network
    time: float = 0.0
    _step_count: int = 0

    def get_step_count(self) -> int:
        return self._step_count

    def get_config(self) -> SimulationConfig:
        return self.config

    def step(self) -> None:
        """
        All or nothing, if step fails, we crash.
        Ideally should be the only function that modifies step_count.
        Fundamental whats-next function in logic computation.
        - compute RSRPs, update values
        - update time-till-last handovers
        - check for handovers
        - perform any handovers if necessary
        - update statistics
        - apply SON tuning if failures detected
        """
        try:
            mobility_update(self.state_space, self.config)
            calculate_rsrp_naive(self.state_space, self.config)
            decide_handovers(
                self.state_space,
                self.config,
                self.handover_policy,
                self.get_step_count(),
            )
            update_timers(self.state_space)

            failure_count_before = (
                self.statistics.early_handover_count
                + self.statistics.late_handover_count
                + self.statistics.ping_pong_handover_count
            )

            for ue in self.state_space.ues:
                if ue.handover_state.handover_this_step:
                    check_handover_type(
                        ue, self.config, self.get_step_count(), True, self.statistics
                    )
                    ue.handover_state.handover_this_step = False
                else:
                    check_handover_type(
                        ue, self.config, self.get_step_count(), False, self.statistics
                    )

            failure_count_after = (
                self.statistics.early_handover_count
                + self.statistics.late_handover_count
                + self.statistics.ping_pong_handover_count
            )

            if failure_count_after > failure_count_before:
                SON.apply_tuning(self.config, self.statistics)

            self._step_count += 1
            self.time = self._step_count * self.config.STEP
        except Exception as e:
            log_error(
                self.get_step_count(),
                "STEP_FAILED",
                exception_type=type(e).__name__,
                exception_message=str(e),
            )
            raise
