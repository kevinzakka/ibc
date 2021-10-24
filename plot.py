"""Generate Figure 4 plot. Pass in --help flag for options."""

import dataclasses
import pathlib
from typing import Dict, Tuple

import dcargs
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull
from tqdm.auto import tqdm

from ibc.experiment import Experiment
from ibc.trainer import TrainState
from train import TrainConfig, make_dataloaders, make_train_state


@dataclasses.dataclass
class Args:
    experiment_name: str
    plot_dir: str = "plots"
    dpi: int = 200


def eval(
    train_state: TrainState,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_coords = []
    mses = []
    for batch in tqdm(dataloaders["test"]):
        input, target = batch
        mean_squared_error = train_state.predict(input, target)
        test_coords.append(target.cpu().numpy())
        mses.append(mean_squared_error.mean(dim=-1).cpu().numpy())

    test_coords = np.concatenate(test_coords)
    train_coords = dataloaders["train"].dataset.coordinates
    mses = np.concatenate(mses)

    return train_coords, test_coords, mses


def plot(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    mses: np.ndarray,
    resolution: Tuple[int, int],
    plot_path: pathlib.Path,
    dpi: int,
) -> None:
    plt.scatter(
        train_coords[:, 0],
        train_coords[:, 1],
        marker="x",
        c="black",
        zorder=2,
        alpha=0.5,
    )
    plt.scatter(
        test_coords[:, 0],
        test_coords[:, 1],
        c=mses,
        cmap="Reds",
        zorder=1,
    )
    plt.colorbar()

    # Find index of predictions with less than 1 pixel error and color them in blue.
    idxs = mses < 1.0
    plt.scatter(
        test_coords[idxs, 0],
        test_coords[idxs, 1],
        marker="o",
        c="blue",
        zorder=1,
        alpha=1.0,
    )

    # Add convext hull of train set.
    hull = ConvexHull(train_coords)
    for simplex in hull.simplices:
        plt.plot(
            train_coords[simplex, 0],
            train_coords[simplex, 1],
            "--",
            zorder=2,
            alpha=0.5,
            c="black",
        )

    plt.xlim(0 - 2, resolution[1] + 2)
    plt.ylim(0 - 2, resolution[0] + 2)

    plt.savefig(plot_path, format="png", dpi=dpi)
    plt.show()


def main(args: Args):
    plot_dir = pathlib.Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    experiment = Experiment(
        identifier=args.experiment_name,
    ).assert_exists()

    # Read saved config file.
    train_config = experiment.read_metadata("config", TrainConfig)

    # Restore training state.
    train_state = make_train_state(train_config)
    experiment.restore_checkpoint(train_state)
    print(f"Loaded checkpoint at step: {train_state.steps}.")

    # Compute MSE for every test set data point.
    dataloaders = make_dataloaders(train_config)
    train_coords, test_coords, mses = eval(train_state, dataloaders)

    # Plot and dump to disk.
    plot(
        train_coords,
        test_coords,
        mses,
        dataloaders["test"].dataset.resolution,
        plot_dir / f"{args.experiment_name}.png",
        args.dpi,
    )


if __name__ == "__main__":
    main(dcargs.parse(Args))
