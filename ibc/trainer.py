from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from .experiment import TensorboardLogData
from .models import ConvMLP, ConvMLPConfig


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0


@dataclasses.dataclass
class TrainState:
    model: ConvMLP
    optimizer: torch.optim.Optimizer
    device: torch.device
    steps: int

    @staticmethod
    def initialize(
        model_config: ConvMLPConfig,
        optim_config: OptimizerConfig,
        device_type: str,
    ) -> TrainState:
        device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = ConvMLP(config=model_config)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
        )

        return TrainState(model=model, optimizer=optimizer, device=device, steps=0)

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
