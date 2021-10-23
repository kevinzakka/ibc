import dataclasses
from typing import Optional, Tuple

import dcargs
import torch
from tqdm.auto import tqdm

from ibc import dataset, models, trainer, utils
from ibc.experiment import Experiment


@dataclasses.dataclass
class TrainConfig:
    experiment_name: str
    seed: int = 0
    device_type: str = "cuda"
    train_dataset_size: int = 30
    test_dataset_size: int = 1000
    max_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    train_batch_size: int = 8
    test_batch_size: int = 64
    spatial_reduction: models.SpatialReduction = models.SpatialReduction.MAX_POOL
    coord_conv: bool = False
    dropout_prob: Optional[float] = None
    num_workers: int = 1
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    log_every_n_steps: int = 10
    checkpoint_every_n_steps: int = 100


def make_dataloaders(
    train_config: TrainConfig,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Initialize train/test dataloaders based on config values."""
    kwargs = {
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
        "shuffle": True,
    }

    train_dataset_config = dataset.DatasetConfig(
        dataset_size=train_config.train_dataset_size,
        seed=train_config.seed,
    )
    train_dataset = dataset.CoordinateRegression(train_dataset_config)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        **kwargs,
    )

    test_dataset_config = dataset.DatasetConfig(
        dataset_size=train_config.test_dataset_size,
        seed=train_config.seed,
    )
    test_dataset = dataset.CoordinateRegression(test_dataset_config)
    test_dataset.exclude(train_dataset.coordinates)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config.test_batch_size,
        **kwargs,
    )

    return train_dataloader, test_dataloader


def make_train_state(train_config: TrainConfig) -> trainer.TrainState:
    """Initialize train state based on config values."""
    in_channels = 3
    if train_config.coord_conv:
        in_channels += 2
    cnn_config = models.CNNConfig(in_channels, [16, 32, 32])

    input_dim = 32
    if train_config.spatial_reduction == models.SpatialReduction.SPATIAL_SOFTMAX:
        input_dim *= 2
    mlp_config = models.MLPConfig(input_dim, 128, 2, 2, train_config.dropout_prob)

    model_config = models.ConvMLPConfig(
        cnn_config=cnn_config,
        mlp_config=mlp_config,
        spatial_reduction=train_config.spatial_reduction,
        coord_conv=train_config.coord_conv,
    )

    optim_config = trainer.OptimizerConfig(
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    train_state = trainer.TrainState.initialize(
        model_config=model_config,
        optim_config=optim_config,
        device_type=train_config.device_type,
    )

    return train_state


def main(train_config: TrainConfig) -> None:
    # Seed RNGs.
    utils.seed_rngs(train_config.seed)

    # CUDA/CUDNN-related shenanigans.
    utils.set_cudnn(train_config.cudnn_deterministic, train_config.cudnn_benchmark)

    experiment = Experiment(identifier=train_config.experiment_name).assert_new()
    experiment.write_metadata("config", train_config)

    train_dataloader, test_dataloader = make_dataloaders(train_config)
    train_state = make_train_state(train_config)

    for epoch in range(train_config.max_epochs):
        if train_state.steps % train_config.checkpoint_every_n_steps == 0:
            experiment.save_checkpoint(train_state.state, step=train_state.steps)

        for batch in tqdm(train_dataloader):
            log_data = train_state.training_step(*batch)

            # Log to tensorboard.
            if train_state.steps % train_config.log_every_n_steps == 0:
                experiment.log(log_data, step=train_state.steps)

    # Save one final checkpoint.
    experiment.save_checkpoint(train_state.state, step=train_state.steps)


if __name__ == "__main__":
    main(dcargs.parse(TrainConfig))
