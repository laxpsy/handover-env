#!/usr/bin/env python3
"""
Compare Q-Learning and REINFORCE agents on the same scenario.
"""
import copy
from algorithms.q_learning import QAgent, QLearningConfig
from algorithms.rl_policy import RLHandoverPolicy
from algorithms.reinforce import REINFORCEAgent
from algorithms.reinforce_policy import REINFORCEHandoverPolicy
from core.reward import compute_reward
from entities.handover_policy import HandoverPolicy
from simulation.config import SimulationConfig
from simulation.scenario_configs import create_network_realistic, REALISTIC_MOBILITY_CONFIG
from simulation.simulator import Simulation
from simulation.statistics import SimulationStatistics

def evaluate_agent(agent_name, policy, config, network, num_episodes=10, steps_per_episode=1000):
    """Evaluate an agent over multiple episodes."""
    stats_list = []
    
    for episode in range(num_episodes):
        statistics = SimulationStatistics()
        sim = Simulation(
            config=config,
            statistics=statistics,
            handover_policy=HandoverPolicy(policy),
            state_space=network,
        )
        sim.reset()
        
        for step in range(steps_per_episode):
            stats_before = copy.copy(sim.statistics)
            sim.step()
            stats_after = sim.statistics
            
            # For REINFORCE, observe reward each step
            if hasattr(policy, 'observe_reward'):
                reward = compute_reward(stats_before, stats_after)
                policy.observe_reward(reward)
        
        # For REINFORCE, finalize episode
        if hasattr(policy, 'end_episode'):
            policy.end_episode()
        
        stats_list.append({
            'episode': episode,
            'early_ho': statistics.early_handover_count,
            'late_ho': statistics.late_handover_count,
            'ping_pong_ho': statistics.ping_pong_handover_count,
            'success_ho': statistics.successful_handover_count,
        })
    
    # Calculate averages
    avg_early = sum(s['early_ho'] for s in stats_list) / len(stats_list)
    avg_late = sum(s['late_ho'] for s in stats_list) / len(stats_list)
    avg_pp = sum(s['ping_pong_ho'] for s in stats_list) / len(stats_list)
    avg_success = sum(s['success_ho'] for s in stats_list) / len(stats_list)
    
    print(f"\n{'='*70}")
    print(f"{agent_name} Performance (avg over {num_episodes} episodes)")
    print(f"{'='*70}")
    print(f"Avg Early Handovers:   {avg_early:6.2f}")
    print(f"Avg Late Handovers:    {avg_late:6.2f}")
    print(f"Avg Ping-Pong HOs:     {avg_pp:6.2f}")
    print(f"Avg Successful HOs:    {avg_success:6.2f}")
    print(f"Total Handovers/Ep:    {avg_early + avg_late + avg_pp + avg_success:6.2f}")
    
    return {
        'early': avg_early,
        'late': avg_late,
        'ping_pong': avg_pp,
        'success': avg_success,
    }

def main():
    # Load trained agents
    print("Loading trained agents...")
    
    # Q-Learning
    ql_config = QLearningConfig()
    ql_agent = QAgent(ql_config)
    ql_agent.load("checkpoints/comparison/ql_realistic_1000ep.pkl")
    ql_agent.config.epsilon = 0.0  # Evaluation mode
    ql_policy = RLHandoverPolicy(ql_agent)
    
    # REINFORCE
    rf_agent = REINFORCEAgent(state_dim=5, hidden_dim=32)
    rf_agent.load("checkpoints/comparison/rf_realistic_1000ep.pt")
    rf_policy = REINFORCEHandoverPolicy(rf_agent, training=False)
    
    # Create network
    config = REALISTIC_MOBILITY_CONFIG
    network = create_network_realistic()
    
    # Evaluate both
    ql_results = evaluate_agent("Q-Learning", ql_policy, config, network, num_episodes=10)
    rf_results = evaluate_agent("REINFORCE", rf_policy, config, network, num_episodes=10)
    
    # Side-by-side comparison
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Q-Learning':>20} {'REINFORCE':>20}")
    print(f"{'-'*70}")
    print(f"{'Early Handovers':<25} {ql_results['early']:>20.2f} {rf_results['early']:>20.2f}")
    print(f"{'Late Handovers':<25} {ql_results['late']:>20.2f} {rf_results['late']:>20.2f}")
    print(f"{'Ping-Pong Handovers':<25} {ql_results['ping_pong']:>20.2f} {rf_results['ping_pong']:>20.2f}")
    print(f"{'Successful Handovers':<25} {ql_results['success']:>20.2f} {rf_results['success']:>20.2f}")
    
    # Winner analysis
    print(f"\n{'='*70}")
    print("WINNER ANALYSIS")
    print(f"{'='*70}")
    
    total_failures_ql = ql_results['early'] + ql_results['late'] + ql_results['ping_pong']
    total_failures_rf = rf_results['early'] + rf_results['late'] + rf_results['ping_pong']
    
    print(f"Total Failures/Episode:")
    print(f"  Q-Learning:  {total_failures_ql:.2f}")
    print(f"  REINFORCE:   {total_failures_rf:.2f}")
    print(f"  Winner: {'Q-Learning' if total_failures_ql < total_failures_rf else 'REINFORCE'}")
    
    print(f"\nAbsolute Success Rate:")
    print(f"  Q-Learning:  {ql_results['success']:.1f}/20 HOs per episode")
    print(f"  REINFORCE:   {rf_results['success']:.1f}/20 HOs per episode")
    print(f"  Winner: {'Q-Learning' if ql_results['success'] > rf_results['success'] else 'REINFORCE'}")

if __name__ == "__main__":
    main()