from __future__ import annotations

import dataclasses
import enum
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from . import experiment, models, optimizers


class TrainStateProtocol(Protocol):
    """Functionality that needs to be implemented by all training states."""

    model: nn.Module
    device: torch.device
    steps: int

    def training_step(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> experiment.TensorboardLogData:
        """Performs a single training step on a mini-batch of data."""

    def evaluate(
        self, dataloader: torch.utils.data.DataLoader
    ) -> experiment.TensorboardLogData:
        """Performs a full evaluation of the model on one epoch."""


@dataclasses.dataclass
class ExplicitTrainState:
    """A feedforward policy trained with a MSE objective."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    steps: int

    @staticmethod
    def initialize(
        model_config: models.ConvMLPConfig,
        optim_config: optimizers.OptimizerConfig,
        device_type: str,
    ) -> ExplicitTrainState:
        device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = models.ConvMLP(config=model_config)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
            betas=(optim_config.beta1, optim_config.beta2),
        )

        return ExplicitTrainState(
            model=model,
            optimizer=optimizer,
            device=device,
            steps=0,
        )

    def training_step(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> experiment.TensorboardLogData:
        self.model.train()

        input = input.to(self.device)
        target = target.to(self.device)

        out = self.model(input)
        loss = F.mse_loss(out, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        return experiment.TensorboardLogData(scalars={"train/loss": loss.item()})

    @torch.no_grad()
    def evaluate(
        self, dataloader: torch.utils.data.DataLoader
    ) -> experiment.TensorboardLogData:
        self.model.eval()

        total_mse = 0.0
        for input, target in tqdm(dataloader, leave=False):
            input = input.to(self.device)
            target = target.to(self.device)

            out = self.model(input)
            mse = F.mse_loss(out, target, reduction="none")
            total_mse += mse.mean(dim=-1).sum().item()

        mean_mse = total_mse / len(dataloader.dataset)
        return experiment.TensorboardLogData(scalars={"test/mse": mean_mse})


@dataclasses.dataclass
class ImplicitTrainState:
    """A conditional EBM trained with an InfoNCE objective."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    stochastic_optimizer: optimizers.StochasticOptimizer
    device: torch.device
    steps: int

    @staticmethod
    def initialize(
        model_config: models.ConvMLPConfig,
        optim_config: optimizers.OptimizerConfig,
        sotchastic_optim_type: optimizers.StochasticOptimizerType,
        device_type: str,
    ) -> ExplicitTrainState:
        device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = models.ConvMLP(config=model_config)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
            betas=(optim_config.beta1, optim_config.beta2),
        )

        stochastic_optimizer = None

        return ImplicitTrainState(
            model=model,
            optimizer=optimizer,
            stochastic_optimizer=stochastic_optimizer,
            device=device,
            steps=0,
        )

    def training_step(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> experiment.TensorboardLogData:
        self.model.train()

        input = input.to(self.device)
        target = target.to(self.device)

        # Generate counter-examples.
        negatives = self.stochastic_optimizer.sample(target)

        out = self.model(input, target, negatives)

        loss = None  # info-nce loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        return experiment.TensorboardLogData(scalars={"train/loss": loss.item()})


class PolicyType(enum.Enum):
    EXPLICIT = ExplicitTrainState
    """An explicit policy is a feedforward structure trained with a MSE objective."""

    IMPLICIT = ImplicitTrainState
    """An implicit policy is a conditional EBM trained with an InfoNCE objective."""
