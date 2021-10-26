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

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """Performs a single inference step on a mini-batch of data."""


@dataclasses.dataclass
class ExplicitTrainState:
    """An explicit feedforward policy trained with a MSE objective."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
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
        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
            betas=(optim_config.beta1, optim_config.beta2),
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=optim_config.lr_scheduler_step,
            gamma=optim_config.lr_scheduler_gamma,
        )

        return ExplicitTrainState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
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
        self.scheduler.step()

        self.steps += 1

        return experiment.TensorboardLogData(
            scalars={
                "train/loss": loss.item(),
                "train/learning_rate": self.scheduler.get_last_lr()[0],
            }
        )

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

    @torch.no_grad()
    def predict(self, input: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(input.to(self.device))


@dataclasses.dataclass
class ImplicitTrainState:
    """An implicit conditional EBM trained with an InfoNCE objective."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    stochastic_optimizer: optimizers.StochasticOptimizer
    device: torch.device
    steps: int

    @staticmethod
    def initialize(
        model_config: models.ConvMLPConfig,
        optim_config: optimizers.OptimizerConfig,
        stochastic_optim_config: optimizers.DerivativeFreeConfig,
        device_type: str,
    ) -> ImplicitTrainState:
        device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = models.EBMConvMLP(config=model_config)
        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
            betas=(optim_config.beta1, optim_config.beta2),
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=optim_config.lr_scheduler_step,
            gamma=optim_config.lr_scheduler_gamma,
        )

        stochastic_optimizer = optimizers.DerivativeFreeOptimizer.initialize(
            stochastic_optim_config,
            device_type,
        )

        return ImplicitTrainState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
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

        # Generate N negatives, one for each element in the batch: (B, N, D).
        negatives = self.stochastic_optimizer.sample(input.size(0), self.model)

        # Merge target and negatives: (B, N+1, D).
        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]

        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self.device)

        # For every element in the mini-batch, there is 1 positive for which the EBM
        # should output a low energy value, and N negatives for which the EBM should
        # output high energy values.
        energy = self.model(input, targets)

        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.steps += 1

        return experiment.TensorboardLogData(
            scalars={
                "train/loss": loss.item(),
                "train/learning_rate": self.scheduler.get_last_lr()[0],
            }
        )

    @torch.no_grad()
    def evaluate(
        self, dataloader: torch.utils.data.DataLoader
    ) -> experiment.TensorboardLogData:
        self.model.eval()

        total_mse = 0.0
        for input, target in tqdm(dataloader, leave=False):
            input = input.to(self.device)
            target = target.to(self.device)

            out = self.stochastic_optimizer.infer(input, self.model)

            mse = F.mse_loss(out, target, reduction="none")
            total_mse += mse.mean(dim=-1).sum().item()

        mean_mse = total_mse / len(dataloader.dataset)
        return experiment.TensorboardLogData(scalars={"test/mse": mean_mse})

    @torch.no_grad()
    def predict(self, input: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.stochastic_optimizer.infer(input.to(self.device), self.model)


class PolicyType(enum.Enum):
    EXPLICIT = ExplicitTrainState
    """An explicit policy is a feedforward structure trained with a MSE objective."""

    IMPLICIT = ImplicitTrainState
    """An implicit policy is a conditional EBM trained with an InfoNCE objective."""
