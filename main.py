import argparse
import copy
import json
import os
from pathlib import Path

from algorithms.handover import naive_handover
from algorithms.q_learning import QAgent, QLearningConfig
from algorithms.rl_policy import RLHandoverPolicy
from algorithms.reinforce import REINFORCEAgent
from algorithms.reinforce_policy import REINFORCEHandoverPolicy
from core.reward import compute_reward
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
        "--mode",
        type=str,
        default="visualize",
        choices=["train", "train-reinforce", "eval", "visualize"],
        help="Execution mode: train (Q-Learning), train-reinforce (Policy Gradient), eval (RL evaluation), visualize (naive baseline)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode. Defaults to 1000.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1200,
        help="Number of training episodes (train mode only). Default: 1200",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to save/load Q-agent checkpoint",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to load pre-trained Q-agent checkpoint",
    )
    parser.add_argument(
        "--run-forever",
        action="store_true",
        help="Run until interrupted (Ctrl+C) or window close (visualize mode only).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Q-Learning learning rate (train mode only). Default: 0.1",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Q-Learning discount factor (train mode only). Default: 0.95",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.996,
        help="Epsilon decay rate per episode (train mode only). Default: 0.996",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for REINFORCE policy network (train-reinforce mode only). Default: 3e-4",
    )
    return parser.parse_args()


def _train_rl_agent(
    config: SimulationConfig,
    network: Network,
    num_episodes: int,
    steps_per_episode: int,
    ql_config: QLearningConfig,
    checkpoint_path: str | None,
    load_checkpoint: str | None,
) -> tuple[QAgent, RLHandoverPolicy, list[dict]]:
    """
    Train Q-Learning agent on handover simulation.
    
    Args:
        config: Simulation config
        network: Network to train on
        num_episodes: Number of episodes
        steps_per_episode: Max steps per episode
        ql_config: Q-Learning hyperparameters
        checkpoint_path: Where to save agent checkpoints
        load_checkpoint: Where to load pre-trained agent from
        
    Returns:
        Tuple of (trained_agent, rl_policy, episode_stats)
    """
    agent = QAgent(ql_config)
    
    # Load pre-trained agent if specified
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f"Loading pre-trained agent from {load_checkpoint}...")
        agent.load(load_checkpoint)
    
    rl_policy = RLHandoverPolicy(agent)
    
    # Create checkpoint directory
    if checkpoint_path:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    episode_stats = []
    
    print(f"\n{'='*70}")
    print(f"Starting Q-Learning Training")
    print(f"Episodes: {num_episodes} | Steps/Episode: {steps_per_episode}")
    print(f"α={ql_config.alpha} | γ={ql_config.gamma} | ε_decay={ql_config.epsilon_decay}")
    print(f"{'='*70}\n")
    
    for episode in range(num_episodes):
        # Create fresh simulation for this episode
        statistics = SimulationStatistics()
        sim = Simulation(
            config=config,
            statistics=statistics,
            handover_policy=HandoverPolicy(rl_policy),
            state_space=Network(ues=[u for u in network.ues], 
                              base_stations=[b for b in network.base_stations]),
        )
        sim.reset()
        
        for step in range(steps_per_episode):
            # Snapshot stats BEFORE step
            stats_before = copy.copy(sim.statistics)
            pre_states = {
                ue.id: agent.discretize_state(ue, sim.state_space)
                for ue in sim.state_space.ues
            }
            
            # Step simulation
            sim.step()
            
            # Compute rewards and update Q-table
            stats_after = sim.statistics
            reward = compute_reward(stats_before, stats_after)
            
            for ue in sim.state_space.ues:
                next_state = agent.discretize_state(ue, sim.state_space)
                rl_policy.observe_reward(ue.id, reward, next_state)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record episode stats
        ep_stats = {
            'episode': episode,
            'epsilon': agent.config.epsilon,
            'q_table_size': len(agent.q_table),
            'early_ho': sim.statistics.early_handover_count,
            'late_ho': sim.statistics.late_handover_count,
            'ping_pong_ho': sim.statistics.ping_pong_handover_count,
            'success_ho': sim.statistics.successful_handover_count,
        }
        episode_stats.append(ep_stats)
        
        # Periodic checkpoint
        if checkpoint_path and (episode + 1) % 50 == 0:
            agent.save(checkpoint_path.replace('.pkl', f'_ep{episode+1}.pkl'))
        
        # Print progress
        if (episode + 1) % 25 == 0:
            print(
                f"Ep {episode+1:3d} | ε={agent.config.epsilon:.4f} | "
                f"Q-states: {len(agent.q_table):4d} | "
                f"Late: {sim.statistics.late_handover_count:2d} | "
                f"PP: {sim.statistics.ping_pong_handover_count:2d} | "
                f"Success: {sim.statistics.successful_handover_count:2d}"
            )
    
    # Final checkpoint
    if checkpoint_path:
        agent.save(checkpoint_path)
        print(f"\nFinal agent saved to {checkpoint_path}")
    
    return agent, rl_policy, episode_stats


def _train_reinforce_agent(
    config: SimulationConfig,
    network: Network,
    num_episodes: int,
    steps_per_episode: int,
    lr: float = 3e-4,
    gamma: float = 0.97,
    checkpoint_path: str | None = None,
    load_checkpoint: str | None = None,
) -> tuple[REINFORCEAgent, REINFORCEHandoverPolicy, list[dict]]:
    """
    Train REINFORCE (policy gradient) agent on handover simulation.
    
    Args:
        config: Simulation config
        network: Network to train on
        num_episodes: Number of episodes
        steps_per_episode: Max steps per episode
        lr: Learningrate for policy network
        gamma: Discount factor
        checkpoint_path: Where to save agent checkpoints
        load_checkpoint: Where to load pre-trained agent from
        
    Returns:
        Tuple of (trained_agent, rl_policy, episode_stats)
    """
    agent = REINFORCEAgent(
        state_dim=5,
        hidden_dim=32,
        lr=lr,
        gamma=gamma,
        baseline=True,
        entropy_coef=0.01,
    )
    
    # Load pre-trained agent if specified
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f"Loading pre-trained REINFORCE agent from {load_checkpoint}...")
        agent.load(load_checkpoint)
    
    rl_policy = REINFORCEHandoverPolicy(agent, training=True)
    
    # Create checkpoint directory
    if checkpoint_path:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    episode_stats = []
    
    print(f"\n{'='*70}")
    print(f"Starting REINFORCE Training (Policy Gradient)")
    print(f"Episodes: {num_episodes} | Steps/Episode: {steps_per_episode}")
    print(f"LR={lr} | γ={gamma} | Entropy Coef=0.01")
    print(f"{'='*70}\n")
    
    for episode in range(num_episodes):
        # Create fresh simulation for this episode
        statistics = SimulationStatistics()
        sim = Simulation(
            config=config,
            statistics=statistics,
            handover_policy=HandoverPolicy(rl_policy),
            state_space=Network(ues=[u for u in network.ues], 
                              base_stations=[b for b in network.base_stations]),
        )
        sim.reset()
        
        for step in range(steps_per_episode):
            # Snapshot stats BEFORE step
            stats_before = copy.copy(sim.statistics)
            
            # Step simulation
            sim.step()
            
            # Compute reward
            stats_after = sim.statistics
            reward = compute_reward(stats_before, stats_after)
            
            # Store reward for all UEs
            rl_policy.observe_reward(reward)
        
        # End of episode: single gradient update over full trajectory
        loss = rl_policy.end_episode()
        
        # Record episode stats
        ep_stats = {
            'episode': episode,
            'loss': loss,
            'early_ho': sim.statistics.early_handover_count,
            'late_ho': sim.statistics.late_handover_count,
            'ping_pong_ho': sim.statistics.ping_pong_handover_count,
            'success_ho': sim.statistics.successful_handover_count,
        }
        episode_stats.append(ep_stats)
        
        # Periodic checkpoint
        if checkpoint_path and (episode + 1) % 200 == 0:
            agent.save(checkpoint_path.replace('.pt', f'_ep{episode+1}.pt'))
        
        # Print progress
        if (episode + 1) % 50 == 0:
            print(
                f"Ep {episode+1:4d} | Loss={loss:8.4f} | "
                f"Late: {sim.statistics.late_handover_count:2d} | "
                f"PP: {sim.statistics.ping_pong_handover_count:3d} | "
                f"Success: {sim.statistics.successful_handover_count:3d}"
            )
    
    # Final checkpoint
    if checkpoint_path:
        agent.save(checkpoint_path)
        print(f"\nFinal agent saved to {checkpoint_path}")
    
    return agent, rl_policy, episode_stats


def _eval_rl_agent(
    config: SimulationConfig,
    network: Network,
    agent: QAgent,
    rl_policy: RLHandoverPolicy,
    num_episodes: int,
    steps_per_episode: int,
) -> list[dict]:
    """
    Evaluate trained Q-Learning agent (epsilon=0, pure exploitation).
    
    Args:
        config: Simulation config
        network: Network to evaluate on
        agent: Trained agent
        rl_policy: RL policy wrapper
        num_episodes: Number of evaluation episodes
        steps_per_episode: Max steps per episode
        
    Returns:
        List of episode statistics
    """
    # Freeze learning (epsilon=0)
    original_epsilon = agent.config.epsilon
    agent.config.epsilon = 0.0
    
    eval_stats = []
    
    print(f"\n{'='*70}")
    print(f"Evaluating Q-Learning Agent (epsilon=0, pure exploitation)")
    print(f"Episodes: {num_episodes} | Steps/Episode: {steps_per_episode}")
    print(f"Q-table size: {len(agent.q_table)}")
    print(f"{'='*70}\n")
    
    for episode in range(num_episodes):
        statistics = SimulationStatistics()
        sim = Simulation(
            config=config,
            statistics=statistics,
            handover_policy=HandoverPolicy(rl_policy),
            state_space=Network(ues=[u for u in network.ues],
                              base_stations=[b for b in network.base_stations]),
        )
        sim.reset()
        
        for step in range(steps_per_episode):
            stats_before = copy.copy(sim.statistics)
            sim.step()
            stats_after = sim.statistics
            reward = compute_reward(stats_before, stats_after)
            
            for ue in sim.state_space.ues:
                next_state = agent.discretize_state(ue, sim.state_space)
                rl_policy.observe_reward(ue.id, reward, next_state)
        
        ep_stats = {
            'episode': episode,
            'early_ho': sim.statistics.early_handover_count,
            'late_ho': sim.statistics.late_handover_count,
            'ping_pong_ho': sim.statistics.ping_pong_handover_count,
            'success_ho': sim.statistics.successful_handover_count,
        }
        eval_stats.append(ep_stats)
        
        if (episode + 1) % 5 == 0:
            print(
                f"Eval Ep {episode+1:2d} | "
                f"Late: {sim.statistics.late_handover_count:2d} | "
                f"PP: {sim.statistics.ping_pong_handover_count:2d} | "
                f"Success: {sim.statistics.successful_handover_count:2d}"
            )
    
    agent.config.epsilon = original_epsilon
    return eval_stats


def _visualize_baseline(
    config: SimulationConfig,
    network: Network,
    max_steps: int | None,
    run_forever: bool,
) -> None:
    """
    Visualize naive handover baseline with Raylib renderer.
    
    Args:
        config: Simulation config
        network: Network to visualize
        max_steps: Max steps to run
        run_forever: Run until window close
    """
    if run_forever and max_steps is not None:
        raise ValueError("Use either --max-steps or --run-forever, not both")

    max_steps_final = None if run_forever else max_steps
    if max_steps_final is None and not run_forever:
        max_steps_final = 1000
    if max_steps_final is not None and max_steps_final <= 0:
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

    print(f"Starting visualization with naive handover policy...")
    if max_steps_final is None:
        print("Running until interrupted (Ctrl+C) or window close")
    else:
        print(f"Running up to {max_steps_final} steps")
    print("Close the window or press ESC to exit")

    step_count = 0
    try:
        while renderer.is_window_open():
            renderer.step()
            simulation.step()
            step_count += 1

            if max_steps_final is not None and step_count >= max_steps_final:
                print("Max steps reached, closing...")
                break
    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C), closing...")
    finally:
        renderer.close()

    print(f"\nVisualization finished after {step_count} steps")
    print(
        f"Statistics: early={statistics.early_handover_count}, "
        f"late={statistics.late_handover_count}, "
        f"ping_pong={statistics.ping_pong_handover_count}, "
        f"success={statistics.successful_handover_count}"
    )


def main():
    args = _parse_args()
    scenario_name = args.scenario
    config, network = _build_scenario(scenario_name)

    max_steps = args.max_steps if args.max_steps else 1000

    print(f"Scenario: {scenario_name}")
    print(f"Mode: {args.mode}")
    
    if args.mode == "visualize":
        _visualize_baseline(config, network, max_steps, args.run_forever)
    
    elif args.mode == "train":
        ql_config = QLearningConfig(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_decay=args.epsilon_decay,
        )
        agent, rl_policy, episode_stats = _train_rl_agent(
            config=config,
            network=network,
            num_episodes=args.episodes,
            steps_per_episode=max_steps,
            ql_config=ql_config,
            checkpoint_path=args.checkpoint,
            load_checkpoint=args.load_checkpoint,
        )
        
        # Print training summary
        print(f"\n{'='*70}")
        print("Training Summary")
        print(f"{'='*70}")
        print(f"Final Q-table size: {len(agent.q_table)}")
        print(f"Final epsilon: {agent.config.epsilon:.4f}")
        
        # Show some example Q-values
        if agent.q_table:
            sample_state = list(agent.q_table.keys())[0]
            print(f"\nExample state {sample_state}:")
            print(f"  Q(stay) = {agent.q_table[sample_state][0]:.4f}")
            print(f"  Q(handover) = {agent.q_table[sample_state][1]:.4f}")
    
    elif args.mode == "eval":
        ql_config = QLearningConfig()
        agent = QAgent(ql_config)
        
        # Load agent
        if args.load_checkpoint and os.path.exists(args.load_checkpoint):
            print(f"Loading pre-trained agent from {args.load_checkpoint}...")
            agent.load(args.load_checkpoint)
        else:
            print("ERROR: For eval mode, must provide --load-checkpoint")
            return
        
        rl_policy = RLHandoverPolicy(agent)
        eval_stats = _eval_rl_agent(
            config=config,
            network=network,
            agent=agent,
            rl_policy=rl_policy,
            num_episodes=10,
            steps_per_episode=max_steps,
        )
        
        # Print evaluation summary
        print(f"\n{'='*70}")
        print("Evaluation Summary (RL Agent)")
        print(f"{'='*70}")
        avg_late = sum(s['late_ho'] for s in eval_stats) / len(eval_stats)
        avg_pp = sum(s['ping_pong_ho'] for s in eval_stats) / len(eval_stats)
        avg_success = sum(s['success_ho'] for s in eval_stats) / len(eval_stats)
        print(f"Avg Late Handovers: {avg_late:.2f}")
        print(f"Avg Ping-Pong Handovers: {avg_pp:.2f}")
        print(f"Avg Successful Handovers: {avg_success:.2f}")
    
    elif args.mode == "train-reinforce":
        agent, rl_policy, episode_stats = _train_reinforce_agent(
            config=config,
            network=network,
            num_episodes=args.episodes,
            steps_per_episode=max_steps,
            lr=getattr(args, 'lr', 3e-4),
            gamma=args.gamma,
            checkpoint_path=args.checkpoint,
            load_checkpoint=args.load_checkpoint,
        )
        
        # Print training summary
        print(f"\n{'='*70}")
        print("REINFORCE Training Summary")
        print(f"{'='*70}")
        print(f"Total episodes trained: {len(episode_stats)}")
        if episode_stats:
            final_ep = episode_stats[-1]
            print(f"Final Episode Loss: {final_ep['loss']:.4f}")
            print(f"Final Late HOs: {final_ep['late_ho']}")
            print(f"Final Ping-Pong HOs: {final_ep['ping_pong_ho']}")
            print(f"Final Success HOs: {final_ep['success_ho']}")


if __name__ == "__main__":
    main()
