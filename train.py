import dataclasses

import dcargs
import torch
from torchkit.utils import seed_rngs
from tqdm.auto import tqdm

from ibc import models
from ibc.dataset import CoordinateRegression, DatasetConfig


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0


@dataclasses.dataclass
class Args:
    model_config: models.ConvMLPConfig = models.ConvMLPConfig(
        cnn_config=models.CNNConfig(3),
        mlp_config=models.MLPConfig(32, 128, 2, 2),
    )
    optim_config: OptimizerConfig = OptimizerConfig()
    dataset_config: DatasetConfig = DatasetConfig()

    seed: int = 0
    device: str = "cuda:0"

    training_steps: int = 1
    print_every_n: int = 100
    batch_size: int = 16
    num_workers: int = 1


@dataclasses.dataclass
class TrainState:
    model: models.ConvMLP
    optimizer: torch.optim.Optimizer
    steps: int

    @staticmethod
    def initialize(
        model_config: models.ConvMLPConfig, optim_config: OptimizerConfig
    ) -> "TrainState":
        model = models.ConvMLP(config=model_config)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
        )
        return TrainState(
            model=model,
            optimizer=optimizer,
            steps=0,
        )


def main(args: Args):
    # Seed RNGs.
    seed_rngs(args.seed)

    # Setup compute device.
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Setup dataset.
    args.dataset_config.seed = args.seed
    dataset = CoordinateRegression(args.dataset_config)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    train_state = TrainState.initialize(args.model_config, args.optim_config)

    for step in tqdm(range(args.training_steps)):
        if step % args.print_every_n == 0:
            pass


if __name__ == "__main__":
    main(dcargs.parse(Args))
