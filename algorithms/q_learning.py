"""
Q-Learning agent for handover policy optimization.

Implements tabular Q-Learning with epsilon-greedy exploration.
State representation is discretized based on RSRP, delta, TTT state, and handover history.
"""

import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import Dict, Tuple

from core.state_encoding import encode_state_discrete
from entities.network import Network
from entities.ue import UE


@dataclass
class QLearningConfig:
    """Configuration for Q-Learning agent."""
    alpha: float = 0.1          # Learning rate
    gamma: float = 0.95         # Discount factor (high: future stability matters)
    epsilon: float = 1.0        # Initial exploration rate
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    bins_rsrp: int = 4          # Discretization buckets for RSRP (4x4x3 state space)
    bins_delta: int = 4         # Discretization buckets for RSRP delta


class QAgent:
    """
    Tabular Q-Learning agent for handover decisions.
    
    State space: (rsrp_bin, delta_bin, ttt_active, steps_since_ho_bin)
    Action space: {0 = stay, 1 = handover to best candidate}
    """

    def __init__(self, config: QLearningConfig):
        """Initialize Q-agent with given configuration."""
        self.config = config
        self.q_table: Dict[Tuple[int, ...], list] = {}  # state_tuple → [Q(stay), Q(handover)]
        self.last_state = None
        self.last_action = None

    def discretize_state(self, ue: UE, network: Network) -> Tuple[int, int, int]:
        """
        Discretize UE state using centralized encoder.
        
        Args:
            ue: User equipment
            network: Network topology
            
        Returns:
            Discrete state tuple (rsrp_bin, delta_bin, ttt_bin)
        """
        return encode_state_discrete(ue, network)

    def select_action(self, state: Tuple[int, ...]) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Discretized state tuple
            
        Returns:
            Action index: 0 (stay) or 1 (handover)
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]

        # Epsilon-greedy exploration
        if np.random.random() < self.config.epsilon:
            return int(np.random.randint(2))
        
        # Exploit: pick action with max Q-value
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...]
    ) -> None:
        """
        Update Q-table using temporal difference learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]

        # Temporal difference target
        td_target = reward + self.config.gamma * max(self.q_table[next_state])
        td_error = td_target - self.q_table[state][action]
        
        # Update Q-value
        self.q_table[state][action] += self.config.alpha * td_error

    def decay_epsilon(self) -> None:
        """Decay exploration rate at end of episode."""
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay
        )

    def save(self, path: str) -> None:
        """Save Q-table and config to disk."""
        with open(path, 'wb') as f:
            pickle.dump(
                {
                    'q_table': self.q_table,
                    'config': self.config,
                },
                f
            )

    def load(self, path: str) -> None:
        """Load Q-table and config from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.config = data['config']

    def get_stats(self) -> dict:
        """Get current training statistics."""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.config.epsilon,
        }
