"""
Reward computation for handover Q-Learning.

Rewards are computed as deltas in simulation statistics to provide
learning signals for the Q-agent. Using deltas (not totals) ensures
the agent is not penalized cumulatively for past mistakes.
"""

from simulation.statistics import SimulationStatistics


def compute_reward(
    stats_before: SimulationStatistics,
    stats_after: SimulationStatistics
) -> float:
    """
    Compute reward based on change in simulation statistics.
    Asymmetric weighting: RLF (late) is 2x worse than ping-pong.
    
    Reward structure:
    - +1.5 for each successful handover (stable)
    - -3.0 for each late handover (RLF event, critical)
    - -1.5 for each ping-pong handover (oscillation)
    
    Ratio of 3.0:1.5 means agent treats 1 RLF as equivalent to 2 ping-pong events.
    This matches 3GPP standards that prioritize network stability over QoS smoothness.
    
    Using DELTAS ensures the agent only receives reward/penalty for
    what changed this step, not cumulatively for all past events.
    
    Args:
        stats_before: Statistics snapshot before simulation step
        stats_after: Statistics snapshot after simulation step
        
    Returns:
        Scalar reward value for the step
    """
    # Compute deltas (changes this step)
    late_delta = (
        stats_after.late_handover_count
        - stats_before.late_handover_count
    )
    pp_delta = (
        stats_after.ping_pong_handover_count
        - stats_before.ping_pong_handover_count
    )
    success_delta = (
        stats_after.successful_handover_count
        - stats_before.successful_handover_count
    )

    # Scaled reward signal to prevent Q-value collapse
    # Weights: +0.5 success, -0.8 late (RLF), -0.2 ping-pong
    # Maintains 4:1 ratio (RLF more critical) while keeping Q-values manageable
    reward = (
        +0.5 * success_delta   # Reward for successful handovers
        - 0.8 * late_delta     # Penalty for RLF (4× ping-pong penalty)
        - 0.2 * pp_delta       # Light penalty for ping-pong (exploration tolerance)
    )

    return reward


def compute_shaped_reward(
    stats_before: SimulationStatistics,
    stats_after: SimulationStatistics,
    ue,
    config,
) -> float:
    """
    Compute reward with potential-based shaping for better learning.
    
    Adds a small bonus for maintaining good serving RSRP to provide
    gradient even when no handover occurs (sparse reward problem).
    
    Args:
        stats_before: Statistics before step
        stats_after: Statistics after step
        ue: User equipment (for RSRP access)
        config: Simulation config
        
    Returns:
        Shaped reward value
    """
    base_reward = compute_reward(stats_before, stats_after)
    
    # Shaping reward: bonus for high serving RSRP (gradient during sparse events)
    serving_rsrp = ue.rsrp.get(ue.serving_bs, -120)
    # Normalize RSRP to [0, 1] range: -120 → 0, -40 → 1
    rsrp_normalized = (serving_rsrp + 120) / 80
    rsrp_normalized = max(0, min(1, rsrp_normalized))  # Clamp to [0,1]
    shaping_bonus = 0.05 * rsrp_normalized  # Increased from 0.01 to 0.05 for stronger gradient
    
    return base_reward + shaping_bonus
