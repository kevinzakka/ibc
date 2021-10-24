from __future__ import annotations

import dataclasses
import enum
from typing import Protocol

import numpy as np
import torch


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999


@dataclasses.dataclass
class StochasticOptimizerConfig:
    bounds: np.ndarray
    """Bounds on the samples, min/max for each dimension."""

    iters: int
    """The total number of optimization iters."""

    samples: int
    """The number of counter-examples to sample per iter."""


class StochasticOptimizer(Protocol):
    """Functionality that needs to be implemented by all stochastic optimizers."""

    def sample(self, target: torch.Tensor) -> torch.Tensor:
        """Sample counter-negatives for training the EBM."""

    def infer(self):
        """Optimize for the best action given a trained EBM."""


@dataclasses.dataclass
class DerivativeFreeOptimizerConfig(StochasticOptimizerConfig):
    sigma_init: float = 0.33
    K: float = 0.5
    iters: int = 3
    samples: int = 2 ** 14


@dataclasses.dataclass
class DerivativeFreeOptimizer:
    def sample(self):
        pass

    def infer(self):
        pass


@dataclasses.dataclass
class StochasticGradientLangevinDynamicsOptimizer:
    def sample(self):
        pass

    def infer(self):
        pass


class StochasticOptimizerType(enum.Enum):
    DERIVATIVE_FREE = DerivativeFreeOptimizer
    LANGEVIN_DYNAMICS = StochasticGradientLangevinDynamicsOptimizer
