import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
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
            raise ValueError(f"'pixel_size' must be odd.")

        self.dataset_size = config.dataset_size
        self.resolution = config.resolution
        self.pixel_size = config.pixel_size
        self.pixel_color = config.pixel_color
        self.seed = config.seed

        self.reset()

    def reset(self) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)

        # Randomly generate pixel coordinates.
        u = np.random.randint(0, self.resolution[0], size=(self.dataset_size))
        v = np.random.randint(0, self.resolution[1], size=(self.dataset_size))

        # Ensure we remain within bounds when we take the pixel size into account.
        slack = self.pixel_size // 2
        u = np.clip(u, a_min=slack, a_max=self.resolution[0] - 1 - slack)
        v = np.clip(v, a_min=slack, a_max=self.resolution[1] - 1 - slack)

        u = u.astype(np.int16)
        v = v.astype(np.int16)

        self._coordinates = np.vstack([u, v]).T

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return self.resolution + (3,)

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

        return image, target


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = CoordinateRegression(DatasetConfig(dataset_size=1_000))

    # Visualize one instance.
    image, target = dataset[np.random.randint(len(dataset))]
    print(target)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()

    # Plot target distribution.
    targets = []
    for _, target in dataset:
        targets.append(target.numpy())
    targets = np.asarray(targets)

    plt.scatter(targets[:, 0], targets[:, 1], marker="x", c="black")
    plt.xlim(0 - 2, dataset.resolution[1] + 2)
    plt.ylim(0 - 2, dataset.resolution[0] + 2)
    plt.show()
