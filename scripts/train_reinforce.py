"""
Training script for REINFORCE agent on handover optimization.

Trains for 1200 episodes on LINEAR scenario with periodic evaluation.
Generates metrics for fair comparison with Q-Learning.
"""

import copy
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.reinforce import REINFORCEAgent
from algorithms.reinforce_policy import REINFORCEHandoverPolicy
from core.reward import compute_reward
from entities.handover_policy import HandoverPolicy
from entities.network import Network
from simulation.scenario_configs import LINEAR_MOBILITY_CONFIG, create_network_linear
from simulation.simulator import Simulation
from simulation.statistics import SimulationStatistics


def train_reinforce():
    """Train REINFORCE agent with periodic evaluation."""

    # ────────────────────────────────────────────────────────────────────────
    # Hyperparameters (matched to Q-Learning for fair comparison)
    # ────────────────────────────────────────────────────────────────────────
    CONFIG = {
        "num_episodes": 1200,
        "steps_per_episode": 1200,
        "eval_every": 50,
        "eval_episodes": 10,
        "state_dim": 5,
        "hidden_dim": 32,
        "lr": 3e-4,
        "gamma": 0.97,  # Match Q-Learning
        "baseline": True,
        "entropy_coef": 0.01,
        "checkpoint_dir": "../checkpoints/",
    }

    print("=" * 70)
    print("REINFORCE Training")
    print("=" * 70)
    print(f"Episodes: {CONFIG['num_episodes']}")
    print(f"Steps/Episode: {CONFIG['steps_per_episode']}")
    print(f"Evaluation: every {CONFIG['eval_every']} eps, {CONFIG['eval_episodes']} episodes each")
    print(f"Learning Rate: {CONFIG['lr']}")
    print(f"Gamma: {CONFIG['gamma']}")
    print(f"Entropy Coef: {CONFIG['entropy_coef']}")
    print("=" * 70 + "\n")

    # ────────────────────────────────────────────────────────────────────────
    # Initialize agent and training infrastructure
    # ────────────────────────────────────────────────────────────────────────
    agent = REINFORCEAgent(
        state_dim=CONFIG["state_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        lr=CONFIG["lr"],
        gamma=CONFIG["gamma"],
        baseline=CONFIG["baseline"],
        entropy_coef=CONFIG["entropy_coef"],
    )

    rl_policy = REINFORCEHandoverPolicy(agent, training=True)
    config = LINEAR_MOBILITY_CONFIG
    network = create_network_linear()

    # Checkpoint directory
    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # Training and evaluation logs
    training_log = []
    evaluation_log = []

    # ────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ────────────────────────────────────────────────────────────────────────
    for episode in range(CONFIG["num_episodes"]):
        # Create fresh simulation for episode
        statistics = SimulationStatistics()
        sim = Simulation(
            config=config,
            statistics=statistics,
            handover_policy=HandoverPolicy(rl_policy),
            state_space=Network(
                ues=[u for u in network.ues],
                base_stations=[b for b in network.base_stations],
            ),
        )
        sim.reset()

        # Run episode steps, collecting rewards
        for step in range(CONFIG["steps_per_episode"]):
            stats_before = copy.copy(sim.statistics)
            sim.step()
            stats_after = sim.statistics
            reward = compute_reward(stats_before, stats_after)
            rl_policy.observe_reward(reward)

        # End of episode: single gradient update
        loss = rl_policy.end_episode()

        training_log.append({
            "episode": episode,
            "loss": loss,
            "late_ho": sim.statistics.late_handover_count,
            "pp_ho": sim.statistics.ping_pong_handover_count,
            "success_ho": sim.statistics.successful_handover_count,
        })

        # ────────────────────────────────────────────────────────────────────
        # Periodic evaluation
        # ────────────────────────────────────────────────────────────────────
        if (episode + 1) % CONFIG["eval_every"] == 0:
            rl_policy.training = False

            eval_metrics = {
                "late_ho": 0,
                "pp_ho": 0,
                "success_ho": 0,
                "early_ho": 0,
            }

            for _ in range(CONFIG["eval_episodes"]):
                statistics = SimulationStatistics()
                sim = Simulation(
                    config=config,
                    statistics=statistics,
                    handover_policy=HandoverPolicy(rl_policy),
                    state_space=Network(
                        ues=[u for u in network.ues],
                        base_stations=[b for b in network.base_stations],
                    ),
                )
                sim.reset()

                for _ in range(CONFIG["steps_per_episode"]):
                    sim.step()

                eval_metrics["late_ho"] += sim.statistics.late_handover_count
                eval_metrics["pp_ho"] += sim.statistics.ping_pong_handover_count
                eval_metrics["success_ho"] += sim.statistics.successful_handover_count
                eval_metrics["early_ho"] += sim.statistics.early_handover_count

            # Average over eval episodes
            for key in eval_metrics:
                eval_metrics[key] /= CONFIG["eval_episodes"]

            evaluation_log.append({
                "episode": episode,
                **eval_metrics,
            })

            # Print progress
            print(
                f"Ep {episode:4d} | Loss={loss:7.4f} | "
                f"PP={eval_metrics['pp_ho']:5.1f} | "
                f"Late={eval_metrics['late_ho']:5.1f} | "
                f"Success={eval_metrics['success_ho']:5.1f}"
            )

            rl_policy.training = True

        # ────────────────────────────────────────────────────────────────────
        # Periodic checkpointing
        # ────────────────────────────────────────────────────────────────────
        if (episode + 1) % 200 == 0:
            checkpoint_path = os.path.join(
                CONFIG["checkpoint_dir"], f"reinforce_ep{episode+1}.pt"
            )
            agent.save(checkpoint_path)

    # ────────────────────────────────────────────────────────────────────────
    # Save final checkpoints and logs
    # ────────────────────────────────────────────────────────────────────────
    final_path = os.path.join(CONFIG["checkpoint_dir"], "reinforce_final.pt")
    agent.save(final_path)
    print(f"\n✓ Final agent saved to {final_path}")

    # Save logs as JSON for analysis
    with open("../logs/reinforce_training.json", "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"✓ Training log saved to ../logs/reinforce_training.json")

    with open("../logs/reinforce_evaluation.json", "w") as f:
        json.dump(evaluation_log, f, indent=2)
    print(f"✓ Evaluation log saved to ../logs/reinforce_evaluation.json")

    # Print final summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    if evaluation_log:
        final_metrics = evaluation_log[-1]
        print(f"Final Ping-Pong:       {final_metrics['pp_ho']:.1f} per episode")
        print(f"Final Late HOs:        {final_metrics['late_ho']:.1f} per episode")
        print(f"Final Successful HOs:  {final_metrics['success_ho']:.1f} per episode")


if __name__ == "__main__":
    train_reinforce()
