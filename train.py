"""Script for training. Pass in --help flag for options."""

import dataclasses
from typing import Dict, Optional

import dcargs
import torch
from tqdm.auto import tqdm

from ibc import dataset, models, trainer, utils
from ibc.experiment import Experiment, TensorboardLogData


@dataclasses.dataclass
class TrainConfig:
    experiment_name: str
    seed: int = 0
    device_type: str = "cuda"
    train_dataset_size: int = 10
    test_dataset_size: int = 500
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    train_batch_size: int = 8
    test_batch_size: int = 64
    policy_type: trainer.PolicyType = trainer.PolicyType.EXPLICIT_MSE
    spatial_reduction: models.SpatialReduction = models.SpatialReduction.SPATIAL_SOFTMAX
    coord_conv: bool = False
    dropout_prob: Optional[float] = None
    num_workers: int = 1
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    log_every_n_steps: int = 10
    checkpoint_every_n_steps: int = 100
    eval_every_n_steps: int = 50


def make_dataloaders(
    train_config: TrainConfig,
) -> Dict[str, torch.utils.data.DataLoader]:
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

    return {
        "train": train_dataloader,
        "test": test_dataloader,
    }


def make_train_state(
    train_config: trainer.AbstractTrainState,
) -> trainer.AbstractTrainState:
    """Initialize train state based on config values."""
    in_channels = 3
    if train_config.coord_conv:
        in_channels += 2
    cnn_config = models.CNNConfig(in_channels, [16, 32, 32])

    input_dim = 32
    if train_config.spatial_reduction == models.SpatialReduction.SPATIAL_SOFTMAX:
        input_dim *= 2
    mlp_config = models.MLPConfig(input_dim, 256, 2, 1, train_config.dropout_prob)

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

    train_state = train_config.policy_type.value.initialize(
        model_config=model_config,
        optim_config=optim_config,
        device_type=train_config.device_type,
    )

    return train_state


def evaluate(
    train_state: trainer.AbstractTrainState,
    test_dataloader: torch.utils.data.DataLoader,
):
    total_mse = 0.0
    for batch in tqdm(test_dataloader, leave=False):
        mse = train_state.predict(*batch, reduction="none")
        total_mse += mse.mean(dim=-1).sum().item()
    num_test = len(test_dataloader.dataset)
    mean_mse = total_mse / num_test
    return TensorboardLogData(scalars={"test/mse": mean_mse})


def main(train_config: TrainConfig) -> None:
    # Seed RNGs.
    utils.seed_rngs(train_config.seed)

    # CUDA/CUDNN-related shenanigans.
    utils.set_cudnn(train_config.cudnn_deterministic, train_config.cudnn_benchmark)

    experiment = Experiment(
        identifier=train_config.experiment_name,
    ).assert_new()

    # Write some metadata.
    experiment.write_metadata("config", train_config)

    train_state = make_train_state(train_config)
    dataloaders = make_dataloaders(train_config)

    for epoch in tqdm(range(train_config.max_epochs)):
        if train_state.steps % train_config.checkpoint_every_n_steps == 0:
            experiment.save_checkpoint(train_state, step=train_state.steps)

        if train_state.steps % train_config.eval_every_n_steps == 0:
            test_log_data = evaluate(train_state, dataloaders["test"])
            experiment.log(test_log_data, step=train_state.steps)

        for batch in dataloaders["train"]:
            train_log_data = train_state.training_step(*batch)

            # Log to tensorboard.
            if train_state.steps % train_config.log_every_n_steps == 0:
                experiment.log(train_log_data, step=train_state.steps)

    # Save one final checkpoint.
    experiment.save_checkpoint(train_state, step=train_state.steps)


if __name__ == "__main__":
    main(dcargs.parse(TrainConfig, description=__doc__))
