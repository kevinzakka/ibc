import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


@dataclasses.dataclass
class DatasetConfig:
    dataset_size: int = 30
    """The size of the dataset. Useful for sample efficiency experiments."""

    resolution: Tuple[int, int] = (100, 100)
    """The resolution of the image."""

    pixel_size: int = 5
    """The size of the pixel whose coordinates we'd like to regress. Must be odd."""

    pixel_color: Tuple[int, int, int] = (0, 255, 0)
    """The color of the pixel whose coordinates we'd like to regress."""

    seed: Optional[int] = None
    """Whether to seed the dataset. Disabled if None."""


class CoordinateRegression(Dataset):
    """Regress the coordinates of a colored pixel block on a white canvas."""

    def __init__(self, config: DatasetConfig) -> None:
        if not config.pixel_size % 2:
            raise ValueError("'pixel_size' must be odd.")

        self.dataset_size = config.dataset_size
        self.resolution = config.resolution
        self.pixel_size = config.pixel_size
        self.pixel_color = config.pixel_color
        self.seed = config.seed
        self.transform = None

        self.reset()

    def reset(self) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)

        self._coordinates = self._sample_coordinates(self.dataset_size)

    def exclude(self, coordinates: np.ndarray) -> None:
        """Exclude the given coordinates, if present, from the previously sampled ones.

        This is useful for ensuring the train set does not accidentally leak into the
        test set.
        """
        mask = (self.coordinates[:, None] == coordinates).all(-1).any(1)
        num_matches = mask.sum()
        while mask.sum() > 0:
            self._coordinates[mask] = self._sample_coordinates(mask.sum())
            mask = (self.coordinates[:, None] == coordinates).all(-1).any(1)
        print(f"Resampled {num_matches} data points.")

    def get_target_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.coordinates.mean(axis=0), self.coordinates.std(axis=0)

    def get_target_bounds(self, percent: float = 0.05) -> np.ndarray:
        """Return per-dimension target min/max plus or minus a small buffer.

        This is described in Section B of the supplemental.
        """
        # Compute per-dimension min and max.
        per_dim_min = self.coordinates.min(axis=0)
        per_dim_max = self.coordinates.max(axis=0)

        # Add a small buffer, typically 0.05 * (max - min).
        buffer = percent * (per_dim_max - per_dim_min)
        bounds = np.vstack([per_dim_min - buffer, per_dim_max + buffer])

        # Clip to allowed min/max.
        slack = self.pixel_size // 2
        bounds[:, 0] = np.clip(
            bounds[:, 0], a_min=slack, a_max=self.resolution[0] - 1 - slack
        )
        bounds[:, 1] = np.clip(
            bounds[:, 1], a_min=slack, a_max=self.resolution[1] - 1 - slack
        )

        # Standardize bounds.
        bounds = (bounds - self.transform[0]) / self.transform[1]

        return bounds

    def set_transform(self, means: np.ndarray, std_devs: np.ndarray) -> None:
        self.transform = (means, std_devs)

    def _sample_coordinates(self, size: int) -> np.ndarray:
        """Helper method for generating pixel coordinates."""
        # Randomly generate pixel coordinates.
        u = np.random.randint(0, self.resolution[0], size=size)
        v = np.random.randint(0, self.resolution[1], size=size)

        # Ensure we remain within bounds when we take the pixel size into account.
        slack = self.pixel_size // 2
        u = np.clip(u, a_min=slack, a_max=self.resolution[0] - 1 - slack)
        v = np.clip(v, a_min=slack, a_max=self.resolution[1] - 1 - slack)

        return np.vstack([u, v]).astype(np.int16).T

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return self.resolution + (3,)

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        uv = self._coordinates[index]

        image = np.full(self.image_shape, fill_value=255, dtype=np.uint8)
        image[
            uv[0] - self.pixel_size // 2 : uv[0] + self.pixel_size // 2 + 1,
            uv[1] - self.pixel_size // 2 : uv[1] + self.pixel_size // 2 + 1,
        ] = self.pixel_color

        image = ToTensor()(image)
        target = torch.as_tensor(uv, dtype=torch.float32)

        if self.transform is not None:
            target = ((target - self.transform[0]) / self.transform[1]).float()

        return image, target


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = CoordinateRegression(DatasetConfig(dataset_size=30))

    # Visualize one instance.
    image, target = dataset[np.random.randint(len(dataset))]
    print(target)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()

    # Plot target distribution.
    targets = dataset.coordinates
    plt.scatter(targets[:, 0], targets[:, 1], marker="x", c="black")
    plt.xlim(0 - 2, dataset.resolution[1] + 2)
    plt.ylim(0 - 2, dataset.resolution[0] + 2)
    plt.show()

    print(f"Target bounds:")
    print(dataset.get_target_bounds())
