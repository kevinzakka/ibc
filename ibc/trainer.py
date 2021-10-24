from __future__ import annotations

import abc
import dataclasses
import enum
from typing import Type, TypeVar

import torch
import torch.nn.functional as F

from .experiment import TensorboardLogData
from .models import ConvMLP, ConvMLPConfig

T = TypeVar("T")


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_b1: float = 0.9
    adam_b2: float = 0.999


@dataclasses.dataclass
class AbstractTrainState(abc.ABC):
    model: ConvMLP
    optimizer: torch.optim.Optimizer
    device: torch.device
    steps: int

    @classmethod
    def initialize(
        cls: Type[T],
        model_config: ConvMLPConfig,
        optim_config: OptimizerConfig,
        device_type: str,
    ) -> T:
        device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = ConvMLP(config=model_config)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
            betas=(optim_config.adam_b1, optim_config.adam_b2),
        )

        return cls(
            model=model,
            optimizer=optimizer,
            device=device,
            steps=0,
        )

    @abc.abstractmethod
    def training_step(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> TensorboardLogData:
        ...

    @torch.no_grad()
    def predict(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        ...


class ExplicitMSETrainState(AbstractTrainState):
    """Uses an explicit feedforward continuous policy trained with MSE."""

    def training_step(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> TensorboardLogData:
        self.model.train()

        input = input.to(self.device)
        target = target.to(self.device)

        out = self.model(input)
        loss = F.mse_loss(out, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        return TensorboardLogData(scalars={"train/loss": loss.item()})

    @torch.no_grad()
    def predict(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        self.model.eval()

        input = input.to(self.device)
        target = target.to(self.device)

        prediction = self.model(input)
        mean_squared_error = F.mse_loss(prediction, target, reduction=reduction)

        return mean_squared_error


class ImplicitEBMTrainState(AbstractTrainState):
    """Uses an implicit conditional EBM trained with an InfoNCE loss."""

    def training_step(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> TensorboardLogData:
        raise NotImplementedError

    @torch.no_grad()
    def predict(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        raise NotImplementedError


class PolicyType(enum.Enum):
    EXPLICIT_MSE = ExplicitMSETrainState
    IMPLICIT_EBM = ImplicitEBMTrainState
