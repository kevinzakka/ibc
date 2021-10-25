"""Generate Figure 4 plot. Pass in --help flag for options."""

import dataclasses
import pathlib
from typing import Dict, Tuple

import dcargs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull
from tqdm.auto import tqdm

from ibc.experiment import Experiment
from ibc.trainer import TrainStateProtocol
from train import TrainConfig, make_dataloaders, make_train_state


@dataclasses.dataclass
class Args:
    experiment_name: str
    plot_dir: str = "assets"
    dpi: int = 200
    threshold: float = 30


def eval(
    train_state: TrainStateProtocol,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_mse = 0.0
    pixel_error = []
    for batch in tqdm(dataloaders["test"]):
        input, target = batch
        prediction = train_state.predict(input).cpu().numpy()
        target = target.cpu().numpy()

        pred_unscaled = np.array(prediction)
        pred_unscaled += 1
        pred_unscaled /= 2
        pred_unscaled[:, 0] *= dataloaders["test"].dataset.resolution[0]
        pred_unscaled[:, 1] *= dataloaders["test"].dataset.resolution[1]

        target_unscaled = np.array(target)
        target_unscaled += 1
        target_unscaled /= 2
        target_unscaled[:, 0] *= dataloaders["test"].dataset.resolution[0]
        target_unscaled[:, 1] *= dataloaders["test"].dataset.resolution[1]

        diff = pred_unscaled - target_unscaled
        pixel_error.append(np.linalg.norm(diff, axis=1))
        total_mse += (diff ** 2).mean(axis=1).sum()

    average_mse = total_mse / len(dataloaders["test"].dataset)
    print(f"Test set MSE: {average_mse}")

    pixel_error = np.concatenate(pixel_error)
    test_coords = dataloaders["test"].dataset.coordinates
    train_coords = dataloaders["train"].dataset.coordinates

    return train_coords, test_coords, pixel_error


def plot(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    errors: np.ndarray,
    resolution: Tuple[int, int],
    plot_path: pathlib.Path,
    dpi: int,
    threshold: float,
) -> None:
    # Threshold the errors so that all generated plot colors cover the same range.
    errors[errors >= threshold] = threshold
    colormap = plt.cm.Reds
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=threshold)

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
        c=errors,
        cmap=colormap,
        norm=normalize,
        zorder=1,
    )
    plt.colorbar()

    # Find index of predictions with less than 1 pixel error and color them in blue.
    idxs = errors < 1.0
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
    plt.close()


def main(args: Args):
    plot_dir = pathlib.Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    experiment = Experiment(
        identifier=args.experiment_name,
    ).assert_exists()

    # Read saved config file.
    train_config = experiment.read_metadata("config", TrainConfig)

    # Restore training state.
    dataloaders = make_dataloaders(train_config)
    train_state = make_train_state(train_config, dataloaders["train"])
    experiment.restore_checkpoint(train_state)
    print(f"Loaded checkpoint at step: {train_state.steps}.")

    # Compute MSE for every test set data point.
    train_coords, test_coords, errors = eval(train_state, dataloaders)

    # Plot and dump to disk.
    plot(
        train_coords,
        test_coords,
        errors,
        dataloaders["test"].dataset.resolution,
        plot_dir / f"{args.experiment_name}.png",
        args.dpi,
        args.threshold,
    )


if __name__ == "__main__":
    main(dcargs.parse(Args))
