from __future__ import annotations

import dataclasses
import enum
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================== #
# Model optimization.
# =================================================================== #


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999


# =================================================================== #
# Stochastic optimization for EBM training and inference.
# =================================================================== #


@dataclasses.dataclass
class StochasticOptimizerConfig:
    bounds: torch.Tensor
    """Bounds on the samples, min/max for each dimension."""

    iters: int
    """The total number of inference iters."""

    train_samples: int
    """The number of counter-examples to sample per iter during training."""

    inference_samples: int
    """The number of candidates to sample per iter during inference."""


class StochasticOptimizer(Protocol):
    """Functionality that needs to be implemented by all stochastic optimizers."""

    device: torch.device

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        """Sample counter-negatives for feeding to the InfoNCE objective."""

    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation."""


@dataclasses.dataclass
class DerivativeFreeConfig(StochasticOptimizerConfig):
    noise_scale: float = 0.33
    noise_shrink: float = 0.5
    iters: int = 3
    train_samples: int = 256
    inference_samples: int = 2 ** 14


@dataclasses.dataclass
class DerivativeFreeOptimizer:
    """A simple derivative-free optimizer."""

    device: torch.device
    noise_scale: float
    noise_shrink: float
    iters: int
    train_samples: int
    inference_samples: int
    bounds: torch.Tensor

    @staticmethod
    def initialize(
        config: DerivativeFreeConfig, device_type: str
    ) -> DerivativeFreeOptimizer:
        return DerivativeFreeOptimizer(
            device=torch.device(device_type if torch.cuda.is_available() else "cpu"),
            noise_scale=config.noise_scale,
            noise_shrink=config.noise_shrink,
            iters=config.iters,
            train_samples=config.train_samples,
            inference_samples=config.inference_samples,
            bounds=config.bounds,
        )

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution."""
        size = (num_samples, self.bounds.shape[1])
        samples = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
        samples = torch.as_tensor(samples, dtype=torch.float32, device=self.device)
        return samples

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        samples = self._sample(batch_size * self.train_samples)
        return samples.reshape(batch_size, self.train_samples, -1)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        # Initialize samples and noise scales.
        noise_scale = self.noise_scale
        samples = self._sample(self.inference_samples)
        return samples
        # for i in range(self.iters):
        #     # Compute energies.
        #     energies = ebm(x, samples)
        #     probs = F.softmax(energies, dim=-1)
        #     # Resample with replacement.
        #     idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
        #     samples = samples[idxs]
        #     # Add noise.
        #     samples = samples + 1
        #     # Clip to target bounds.
        #     noise_scale *= self.noise_shrink
        # Return target with highest probability.


class StochasticOptimizerType(enum.Enum):
    # Note(kevin): The paper describes three types of samplers. Right now, we just have
    # the derivative free sampler implemented.
    DERIVATIVE_FREE = enum.auto()


if __name__ == "__main__":
    from dataset import CoordinateRegression, DatasetConfig

    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()

    config = DerivativeFreeConfig(bounds=bounds, train_samples=256)
    so = DerivativeFreeOptimizer.initialize(config, "cuda")

    negatives = so.sample(64, nn.Identity())
    assert negatives.shape == (64, config.train_samples, bounds.shape[1])
