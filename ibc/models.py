import dataclasses
import enum
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchkit.layers import GlobalAvgPool2d, SpatialSoftArgmax


class ActivationType(enum.Enum):
    RELU = partial(nn.ReLU, inplace=False)
    SELU = partial(nn.SiLU, inplace=False)


@dataclasses.dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    hidden_depth: int
    dropout_prob: Optional[float] = None
    activation_fn: ActivationType = ActivationType.RELU


class MLP(nn.Module):
    """A feedforward multi-layer perceptron."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        if config.dropout_prob is not None:
            dropout_layer = partial(nn.Dropout, p=config.dropout_prob)
        else:
            dropout_layer = nn.Identity

        if config.hidden_depth == 0:
            layers = [nn.Linear(config.input_dim, config.output_dim)]
        else:
            layers = [
                nn.Linear(config.input_dim, config.hidden_dim),
                config.activation_fn.value(),
                dropout_layer(),
            ]
            for _ in range(config.hidden_depth - 1):
                layers += [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    config.activation_fn.value(),
                    dropout_layer(),
                ]
            layers += [nn.Linear(config.hidden_dim, config.output_dim)]
        layers = [l for l in layers if not isinstance(l, nn.Identity)]

        self.net = nn.Sequential(*layers)

        # Weight initialization.
        def weight_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity=config.activation_fn.name.lower(),
                )
                nn.init.constant_(m.bias, 0.0)

        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        activation_fn: ActivationType = ActivationType.RELU,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)
        self.activation = activation_fn.value()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(x)
        out = self.conv2(out)
        return out + x


@dataclasses.dataclass(frozen=True)
class CNNConfig:
    in_channels: int
    blocks: Tuple[int, ...] = dataclasses.field(default=(16, 32, 32))
    activation_fn: ActivationType = ActivationType.RELU


class CNN(nn.Module):
    """A residual convolutional network."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()

        depth_in = config.in_channels

        layers = []
        for depth_out in config.blocks:
            layers.extend(
                [
                    nn.Conv2d(depth_in, depth_out, 3, padding=1),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    ResidualBlock(depth_out, config.activation_fn),
                    ResidualBlock(depth_out, config.activation_fn),
                ]
            )
            depth_in = depth_out

        self.net = nn.Sequential(*layers)
        self.activation = config.activation_fn.value()

        # Weight initialization.
        def weight_init(m: nn.Module) -> None:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity=config.activation_fn.name.lower(),
                )
                nn.init.constant_(m.bias, 0.0)

        self.apply(weight_init)

    def forward(self, x: torch.Tensor, activate: bool = False) -> torch.Tensor:
        out = self.net(x)
        if activate:
            return self.activation(out)
        return out


class CoordConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, image_height, image_width = x.size()
        y_coords = (
            2.0
            * torch.arange(image_height).unsqueeze(1).expand(image_height, image_width)
            / (image_height - 1.0)
            - 1.0
        )
        x_coords = (
            2.0
            * torch.arange(image_width).unsqueeze(0).expand(image_height, image_width)
            / (image_width - 1.0)
            - 1.0
        )
        coords = torch.stack((y_coords, x_coords), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        x = torch.cat((coords.to(x.device), x), dim=1)
        return x


class SpatialReduction(enum.Enum):
    SPATIAL_SOFTMAX = SpatialSoftArgmax
    AVERAGE_POOL = GlobalAvgPool2d


@dataclasses.dataclass(frozen=True)
class ConvMLPConfig:
    cnn_config: CNNConfig
    mlp_config: MLPConfig
    spatial_reduction: SpatialReduction = SpatialReduction.AVERAGE_POOL
    coord_conv: bool = False


class ConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        out = self.reducer(out)
        out = self.mlp(out)
        return out


if __name__ == "__main__":
    config = ConvMLPConfig(
        cnn_config=CNNConfig(5),
        mlp_config=MLPConfig(32, 128, 2, 2),
        spatial_reduction=SpatialReduction.AVERAGE_POOL,
        coord_conv=True,
    )

    net = ConvMLP(config)

    x = torch.randn(2, 3, 100, 100)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (2, 2)
