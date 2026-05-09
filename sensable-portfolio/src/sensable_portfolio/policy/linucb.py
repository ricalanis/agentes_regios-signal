"""Disjoint LinUCB via MABWiser, Gaussian reward in [-1, 1]."""
from __future__ import annotations

import pickle
from typing import Iterable

import numpy as np
from mabwiser.mab import MAB, LearningPolicy


class LinUCBPolicy:
    def __init__(self, arms: Iterable[str], context_dim: int, alpha: float = 1.0):
        self._arms = list(arms)
        self._dim = context_dim
        self._alpha = alpha
        self._mab = MAB(arms=self._arms, learning_policy=LearningPolicy.LinUCB(alpha=alpha))
        # MABWiser requires an initial fit — seed with one zero observation per arm
        zeros = np.zeros((len(self._arms), context_dim))
        decisions = list(self._arms)
        rewards = [0.0] * len(self._arms)
        self._mab.fit(decisions=decisions, rewards=rewards, contexts=zeros)

    def predict(self, context: np.ndarray) -> str:
        result = self._mab.predict(contexts=context.reshape(1, -1))
        # MABWiser may return a list or a single value depending on version
        if isinstance(result, list):
            return result[0]
        return result

    def partial_fit(self, context: np.ndarray, arm: str, reward: float) -> None:
        self._mab.partial_fit(decisions=[arm], rewards=[float(reward)],
                              contexts=context.reshape(1, -1))

    def add_arm(self, arm: str) -> None:
        if arm in self._arms:
            return
        self._arms.append(arm)
        self._mab.add_arm(arm)
        self._mab.partial_fit(decisions=[arm], rewards=[0.0],
                              contexts=np.zeros((1, self._dim)))

    def snapshot(self) -> bytes:
        return pickle.dumps({"mab": self._mab, "arms": self._arms,
                             "dim": self._dim, "alpha": self._alpha})

    def restore(self, blob: bytes) -> None:
        state = pickle.loads(blob)
        self._mab = state["mab"]
        self._arms = state["arms"]
        self._dim = state["dim"]
        self._alpha = state["alpha"]
