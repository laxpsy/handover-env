from simulation.config import SimulationConfig
from simulation.statistics import SimulationStatistics


class SON:
    HYSTERISIS_STEP: float = 1.0
    TTT_STEP: float = 1.0
    MIN_HYSTERISIS: float = 0.0
    MAX_HYSTERISIS: float = 10.0
    MIN_TTT: float = 1.0
    MAX_TTT: float = 20.0

    @staticmethod
    def tune_parameters(
        config: SimulationConfig, statistics: SimulationStatistics
    ) -> tuple[float, float]:
        total_handovers = (
            statistics.early_handover_count
            + statistics.late_handover_count
            + statistics.ping_pong_handover_count
        )

        if total_handovers == 0:
            return config.HYSTERISIS_MARGIN, config.TIME_TO_TRIGGER

        early_ratio = statistics.early_handover_count / total_handovers
        late_ratio = statistics.late_handover_count / total_handovers
        ping_pong_ratio = statistics.ping_pong_handover_count / total_handovers

        new_hysterisis = config.HYSTERISIS_MARGIN
        new_ttt = config.TIME_TO_TRIGGER

        if ping_pong_ratio > 0.3:
            new_hysterisis = min(
                config.HYSTERISIS_MARGIN + SON.HYSTERISIS_STEP,
                SON.MAX_HYSTERISIS,
            )
        elif ping_pong_ratio < 0.1:
            new_hysterisis = max(
                config.HYSTERISIS_MARGIN - SON.HYSTERISIS_STEP * 0.5,
                SON.MIN_HYSTERISIS,
            )

        if early_ratio > 0.3:
            new_ttt = min(
                config.TIME_TO_TRIGGER + SON.TTT_STEP,
                SON.MAX_TTT,
            )
        elif late_ratio > 0.3:
            new_ttt = max(
                config.TIME_TO_TRIGGER - SON.TTT_STEP,
                SON.MIN_TTT,
            )

        return new_hysterisis, new_ttt

    @staticmethod
    def apply_tuning(
        config: SimulationConfig, statistics: SimulationStatistics
    ) -> None:
        new_hysterisis, new_ttt = SON.tune_parameters(config, statistics)
        config.HYSTERISIS_MARGIN = new_hysterisis
        config.TIME_TO_TRIGGER = new_ttt
