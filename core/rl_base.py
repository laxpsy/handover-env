"""
Base interfaces for RL agents and policy wrappers.
"""
from abc import ABC, abstractmethod
from typing import Any, Tuple

class BaseRLAgent(ABC):
    @abstractmethod
    def select_action(self, state: Any) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def observe_reward(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self) -> float:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

class RLHandoverPolicyBase(ABC):
    @abstractmethod
    def __call__(self, ue, network, config, step_count):
        pass

    @abstractmethod
    def observe_reward(self, reward: float):
        pass

    @abstractmethod
    def end_episode(self) -> float:
        pass
