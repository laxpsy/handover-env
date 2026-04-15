import argparse

from algorithms.handover import naive_handover
from entities.handover import UEHandoverState
from entities.handover_policy import HandoverPolicy
from entities.network import Network
from entities.rendering_entities import Coordinates
from entities.ue import UE, UEMovementType
from rendering import Renderer, RendererConfig
from simulation.scenario_configs import (
    LINEAR_MOBILITY_CONFIG,
    REALISTIC_MOBILITY_CONFIG,
    create_network_linear,
    create_network_realistic,
)
from simulation.config import SimulationConfig
from simulation.simulator import Simulation
from simulation.statistics import SimulationStatistics


def create_test_network(
    config: SimulationConfig,
    num_ues: int = 2,
    num_bs: int = 3,
) -> Network:
    from entities.base_station import BaseStation

    ues = []
    for i in range(num_ues):
        ue = UE(
            id=i,
            coordinates=Coordinates(
                x=100.0 + i * 50,
                y=100.0 + i * 50,
            ),
            velocity_x=1.0,
            velocity_y=0.5,
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
    for i in range(num_bs):
        bs = BaseStation(
            id=i,
            coordinates=Coordinates(
                x=150.0 + i * 250,
                y=300.0,
            ),
            tx_power=config.DEFAULT_TX_POWER,
            transmission_frequency=config.DEFAULT_FREQUENCY,
        )
        base_stations.append(bs)

    return Network(ues=ues, base_stations=base_stations)


def _build_scenario(scenario_name: str) -> tuple[object, Network]:
    normalized = scenario_name.strip().lower()
    if normalized == "linear":
        return LINEAR_MOBILITY_CONFIG, create_network_linear()
    if normalized == "realistic":
        return REALISTIC_MOBILITY_CONFIG, create_network_realistic()
    if normalized == "test":
        config = SimulationConfig(
            SCREEN_WIDTH=800,
            SCREEN_HEIGHT=600,
            STEP=100,
            HYSTERISIS_MARGIN=3,
            TIME_TO_TRIGGER=3,
            RLF_FAILURE_THRESHOLD=-97,
            DEFAULT_TX_POWER=53.0,
            DEFAULT_FREQUENCY=24.25e9,
        )
        return config, create_test_network(config, num_ues=2, num_bs=3)
    raise ValueError(
        f"Unknown scenario '{scenario_name}'. Use: linear, realistic, or test"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run handover simulation")
    parser.add_argument(
        "scenario",
        nargs="?",
        default="realistic",
        choices=["linear", "realistic", "test"],
        help="Scenario preset to run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps to run. Defaults to 1000.",
    )
    parser.add_argument(
        "--run-forever",
        action="store_true",
        help="Run until interrupted (Ctrl+C) or window close.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    scenario_name = args.scenario
    config, network = _build_scenario(scenario_name)

    if args.run_forever and args.max_steps is not None:
        raise ValueError("Use either --max-steps or --run-forever, not both")

    max_steps = None if args.run_forever else args.max_steps
    if max_steps is None and not args.run_forever:
        max_steps = 1000
    if max_steps is not None and max_steps <= 0:
        raise ValueError("--max-steps must be a positive integer")

    statistics = SimulationStatistics()
    policy = HandoverPolicy(naive_handover)

    simulation = Simulation(
        config=config,
        statistics=statistics,
        handover_policy=policy,
        state_space=network,
    )

    renderer_config = RendererConfig(
        fps=60,
        use_smooth_positions=True,
    )
    renderer = Renderer(simulation, renderer_config, "Handover Simulation Demo")

    print(f"Starting simulation with Raylib renderer ({scenario_name})...")
    if max_steps is None:
        print("Running until interrupted (Ctrl+C) or window close")
    else:
        print(f"Running up to {max_steps} steps")
    print("Close the window or press ESC to exit")

    step_count = 0
    try:
        while renderer.is_window_open():
            renderer.step()
            simulation.step()
            step_count += 1

            if max_steps is not None and step_count >= max_steps:
                print("Max steps reached, closing...")
                break
    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C), closing...")
    finally:
        renderer.close()

    print(f"Simulation finished after {step_count} steps")
    print(
        f"Statistics: early={statistics.early_handover_count}, "
        f"late={statistics.late_handover_count}, "
        f"ping_pong={statistics.ping_pong_handover_count}, "
        f"success={statistics.successful_handover_count}"
    )


if __name__ == "__main__":
    main()
