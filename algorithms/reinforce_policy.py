"""
REINFORCE-based handover policy wrapper.

Adapts REINFORCEAgent to conform to the HandoverPolicy callable interface.
Manages continuous state encoding and per-UE trajectory tracking.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple

from algorithms.reinforce import REINFORCEAgent
from core.state_encoding import encode_state_continuous
from entities.network import Network
from entities.ue import UE
from simulation.config import SimulationConfig


class REINFORCEHandoverPolicy:
    """
    Wraps REINFORCEAgent to conform to HandoverPolicy interface.
    
    Manages:
    - Continuous state encoding (no binning)
    - Per-UE action selection and trajectory storage
    - Reward propagation to agent
    """

    def __init__(self, agent: REINFORCEAgent, training: bool = True):
        """
        Initialize REINFORCE policy.
        
        Args:
            agent: REINFORCEAgent instance
            training: If True, sample actions; if False, use greedy argmax
        """
        self.agent = agent
        self.training = training
        # Per-UE pending buffers: ue_id → (log_prob, entropy)
        self._pending: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(
        self,
        ue: UE,
        network: Network,
        config: SimulationConfig,
        step_count: int,
    ) -> Optional[int]:
        """
        Make handover decision for a UE.
        
        Args:
            ue: User equipment making decision
            network: Current network topology
            config: Simulation config
            step_count: Current simulation step
            
        Returns:
            Target BS ID (handover) or None (stay)
        """
        # Encode state continuously (no discretization)
        state = encode_state_continuous(ue, network)

        if self.training:
            # Training: sample from policy, store trajectory components
            action, log_prob, entropy = self.agent.select_action(state)
            self._pending[ue.id] = (log_prob, entropy)
        else:
            # Evaluation: greedy policy (deterministic argmax)
            x = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits = self.agent.policy(x)
                action = logits.argmax().item()

        if action == 0:
            return None  # Stay with current BS

        # Action 1: Handover to best candidate
        candidate_rsrps = {
            bs_id: rsrp
            for bs_id, rsrp in ue.rsrp.items()
            if bs_id != ue.serving_bs
        }
        if not candidate_rsrps:
            return None

        best_bs = max(candidate_rsrps, key=candidate_rsrps.get)
        return best_bs

    def observe_reward(self, reward: float):
        """
        Propagate reward to all UEs that acted this step.
        Called once per simulation step with shared reward signal.
        
        Args:
            reward: Reward value from this step
        """
        for ue_id, (log_prob, entropy) in self._pending.items():
            self.agent.store_transition(log_prob, reward, entropy)
        self._pending.clear()

    def end_episode(self) -> float:
        """
        Finalize episode: perform REINFORCE gradient update.
        Called once per episode after all steps complete.
        
        Returns:
            Loss value for logging
        """
        loss = self.agent.update()
        return loss
