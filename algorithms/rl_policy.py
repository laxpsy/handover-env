"""
RL-based handover policy that wraps a Q-Learning agent.

Conforms to the HandoverPolicy interface while managing state transitions
and reward propagation from the simulator.
"""

from typing import Dict, Tuple, Optional

from algorithms.q_learning import QAgent
from entities.network import Network
from entities.ue import UE
from simulation.config import SimulationConfig


class RLHandoverPolicy:
    """
    Wraps QAgent to conform to the HandoverPolicy callable interface.
    
    Manages:
    - State discretization per UE
    - Action selection (stay vs handover to best candidate)
    - Reward observation and Q-table updates
    """

    def __init__(self, agent: QAgent):
        """
        Initialize RL policy wrapper.
        
        Args:
            agent: QAgent instance to use for decision-making
        """
        self.agent = agent
        # Track pending (state, action) pairs awaiting reward
        self._pending: Dict[int, Tuple[Tuple[int, ...], int]] = {}

    def __call__(
        self,
        ue: UE,
        network: Network,
        config: SimulationConfig,
        step_count: int
    ) -> Optional[int]:
        """
        Make handover decision for a UE.
        
        This method is called by the simulator's handover decision logic.
        It returns the ID of the target base station, or None to stay.
        
        state = encode_state_discrete(ue, network)
            ue: User equipment to make decision for
            network: Network state
            config: Simulation config
            step_count: Current step number
            
        Returns:
            Target BS ID to handover to, or None to stay
        """
        # Get current state
        state = self.agent.discretize_state(ue, network)
        
        # Select action
        action = self.agent.select_action(state)
        
        # Store pending update for later reward observation
        self._pending[ue.id] = (state, action)

        if action == 0:
            # Action 0: Stay (no handover)
            return None

        # Action 1: Handover to best candidate
        candidate = self._get_best_candidate(ue)
        return candidate

    def observe_reward(
        self,
        ue_id: int,
        reward: float,
        next_state: Tuple[int, ...]
    ) -> None:
        """
        Process reward and update Q-table for a UE after a step.
        
        Called by the training loop after step() completes and
        reward has been computed from statistics delta.
        
        Args:
            ue_id: UE ID that received this reward
            reward: Scalar reward value
            next_state: Next discretized state
        """
        if ue_id in self._pending:
            state, action = self._pending.pop(ue_id)
            self.agent.update(state, action, reward, next_state)

    def _get_best_candidate(self, ue: UE) -> Optional[int]:
        """
        Find best candidate base station (not serving BS).
        
        Args:
            ue: UE to find candidate for
            
        Returns:
            Best candidate BS ID, or None if no candidates exist
        """
        candidates = {
            bs_id: rsrp
            for bs_id, rsrp in ue.rsrp.items()
            if bs_id != ue.serving_bs
        }
        
        if not candidates:
            return None
        
        best_id = max(candidates.keys(), key=lambda bs_id: candidates[bs_id])
        return best_id

    def get_pending_count(self) -> int:
        """Get number of UEs awaiting reward observation."""
        return len(self._pending)
