"""Script for training. Pass in --help flag for options."""

import dataclasses
from typing import Dict, Optional

import dcargs
import torch
from tqdm.auto import tqdm

from ibc import dataset, models, optimizers, trainer, utils
from ibc.experiment import Experiment


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
    spatial_reduction: models.SpatialReduction = models.SpatialReduction.SPATIAL_SOFTMAX
    coord_conv: bool = False
    dropout_prob: Optional[float] = None
    num_workers: int = 1
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    log_every_n_steps: int = 10
    checkpoint_every_n_steps: int = 100
    eval_every_n_steps: int = 200
    target_bounds_percent: float = 0.05
    policy_type: trainer.PolicyType = trainer.PolicyType.EXPLICIT


def make_dataloaders(
    train_config: TrainConfig,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Initialize train/test dataloaders based on config values."""
    kwargs = {"num_workers": 0, "pin_memory": torch.cuda.is_available()}

    # Train split.
    train_dataset_config = dataset.DatasetConfig(
        dataset_size=train_config.train_dataset_size,
        seed=train_config.seed,
    )
    train_dataset = dataset.CoordinateRegression(train_dataset_config)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        shuffle=True,
        **kwargs,
    )

    # Compute train set target mean/std-dev.
    train_target_mean, train_target_std = train_dataset.get_target_statistics()
    train_dataset.set_transform(train_target_mean, train_target_std)

    # Test split.
    test_dataset_config = dataset.DatasetConfig(
        dataset_size=train_config.test_dataset_size,
        seed=train_config.seed,
    )
    test_dataset = dataset.CoordinateRegression(test_dataset_config)
    test_dataset.set_transform(train_target_mean, train_target_std)
    test_dataset.exclude(train_dataset.coordinates)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config.test_batch_size,
        shuffle=False,
        **kwargs,
    )

    return {
        "train": train_dataloader,
        "test": test_dataloader,
    }


def make_train_state(
    train_config: trainer.TrainStateProtocol,
    train_dataloader: torch.utils.data.DataLoader,
) -> trainer.TrainStateProtocol:
    """Initialize train state based on config values."""
    in_channels = 3
    if train_config.coord_conv:
        in_channels += 2
    cnn_config = models.CNNConfig(in_channels, [16, 32, 32])

    input_dim = 32
    output_dim = 2
    if train_config.spatial_reduction == models.SpatialReduction.SPATIAL_SOFTMAX:
        input_dim *= 2
    if train_config.policy_type == trainer.PolicyType.IMPLICIT:
        input_dim += 2  # Dimension of the targets.
        output_dim = 1
    mlp_config = models.MLPConfig(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim,
        hidden_depth=1,
        dropout_prob=train_config.dropout_prob,
    )

    model_config = models.ConvMLPConfig(
        cnn_config=cnn_config,
        mlp_config=mlp_config,
        spatial_reduction=train_config.spatial_reduction,
        coord_conv=train_config.coord_conv,
    )

    optim_config = optimizers.OptimizerConfig(
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    if train_config.policy_type == trainer.PolicyType.EXPLICIT:
        train_state = trainer.ExplicitTrainState.initialize(
            model_config=model_config,
            optim_config=optim_config,
            device_type=train_config.device_type,
        )
    else:
        # Compute bounds on target values in the training data.
        target_bounds = train_dataloader.dataset.get_target_bounds(
            train_config.target_bounds_percent
        )

        stochastic_optim_config = optimizers.DerivativeFreeConfig(bounds=target_bounds)

        train_state = trainer.ImplicitTrainState.initialize(
            model_config=model_config,
            optim_config=optim_config,
            stochastic_optim_config=stochastic_optim_config,
            device_type=train_config.device_type,
        )

    return train_state


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

    # Initialize train and test dataloaders.
    dataloaders = make_dataloaders(train_config)

    train_state = make_train_state(train_config, dataloaders["train"])

    for epoch in tqdm(range(train_config.max_epochs)):
        if not train_state.steps % train_config.checkpoint_every_n_steps:
            experiment.save_checkpoint(train_state, step=train_state.steps)

        if not train_state.steps % train_config.eval_every_n_steps:
            test_log_data = train_state.evaluate(dataloaders["test"])
            experiment.log(test_log_data, step=train_state.steps)

        for batch in dataloaders["train"]:
            train_log_data = train_state.training_step(*batch)

            # Log to tensorboard.
            if not train_state.steps % train_config.log_every_n_steps:
                experiment.log(train_log_data, step=train_state.steps)

    # Save one final checkpoint.
    experiment.save_checkpoint(train_state, step=train_state.steps)


if __name__ == "__main__":
    main(dcargs.parse(TrainConfig, description=__doc__))
