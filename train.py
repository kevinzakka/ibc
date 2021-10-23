import dataclasses

import dcargs
import torch
from tqdm.auto import tqdm

from ibc import dataset, models, trainer, utils
from ibc.experiment import Experiment


@dataclasses.dataclass
class TrainConfig:
    experiment_name: str
    seed: int = 0

    dataset_config: dataset.DatasetConfig = dataset.DatasetConfig(
        dataset_size=30,
        resolution=(100, 100),
        pixel_size=5,
        seed=0,
    )

    max_epochs: int = 10
    batch_size: int = 8

    num_workers: int = 1

    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    log_every_n_steps: int = 10
    checkpoint_every_n_steps: int = 1


def main(train_config: TrainConfig) -> None:
    # Seed RNGs.
    utils.seed_rngs(train_config.seed)

    # CUDA/CUDNN-related shenanigans.
    utils.set_cudnn(train_config.cudnn_deterministic, train_config.cudnn_benchmark)

    # Setup compute device.
    cuda_available: bool = torch.cuda.is_available()
    if cuda_available:
        device = torch.device(train_config.device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize dataset.
    train_dataloader = torch.utils.data.DataLoader(
        dataset.CoordinateRegression(train_config.dataset_config),
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers if cuda_available else 0,
        pin_memory=cuda_available,
        shuffle=True,
        drop_last=True,
    )

    experiment = Experiment(identifier=train_config.experiment_name).assert_new()

    train_state = trainer.TrainState.initialize(
        model_config=models.ConvMLPConfig(
            cnn_config=models.CNNConfig(3, [16, 32, 32]),
            mlp_config=models.MLPConfig(32, 128, 2, 2),
            spatial_reduction=models.SpatialReduction.AVERAGE_POOL,
            coord_conv=False,
        ),
        optim_config=trainer.OptimizerConfig(),
        device=device,
    )

    for epoch in range(train_config.max_epochs):
        if train_state.steps % train_config.checkpoint_every_n_steps == 0:
            experiment.save_checkpoint(train_state.state, step=train_state.steps)

        for batch in tqdm(train_dataloader):
            input, target = batch
            log_data = train_state.training_step(input, target)

            # Log to tensorboard.
            if train_state.steps % train_config.log_every_n_steps == 0:
                experiment.log(log_data, step=train_state.steps)

    # Save one final checkpoint.
    experiment.save_checkpoint(train_state.state, step=train_state.steps)


if __name__ == "__main__":
    main(dcargs.parse(TrainConfig))
