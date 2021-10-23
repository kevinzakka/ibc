import dataclasses
from typing import Any, Dict

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
    ) -> "TrainState":
        device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = ConvMLP(config=model_config)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
        )

        return TrainState(model=model, optimizer=optimizer, device=device, steps=0)

    @property
    def state(self) -> Dict[str, Any]:
        """Returns all objects with a state_dict attribute."""
        state: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if hasattr(v, "state_dict"):
                state[k] = v
        return state

    def training_step(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> TensorboardLogData:
        self.model.train()

        input = input.to(self.device)
        target = target.to(self.device)

        out = self.model(input)
        loss = F.mse_loss(out, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        return TensorboardLogData(scalars={"train/loss": loss.item()})

    @torch.inference_mode()
    def predict(self, input: torch.Tensor, target: torch.Tensor):
        self.model.eval()
