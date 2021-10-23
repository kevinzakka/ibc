import dataclasses
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


@dataclasses.dataclass
class DatasetConfig:
    dataset_size: int = 30
    """The size of the dataset. Useful for sample efficiency experiments."""

    resolution: Tuple[int, int] = dataclasses.field(default=(100, 100))
    """The resolution of the image."""

    pixel_size: int = 5
    """The size of the pixel whose coordinates we'd like to regress."""

    pixel_color: Tuple[int, int, int] = dataclasses.field(default=(0, 255, 0))
    """The color of the pixel whose coordinates we'd like to regress."""

    seed: int = None
    """Whether to seed the dataset. Does not seed if None."""


class CoordinateRegression(Dataset):
    """A coordinate regression task.

    You are given a white image with a colored square, and the goal is to regress the
    pixel coordinates of this square.
    """

    def __init__(self, config: DatasetConfig) -> None:
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
        u = np.clip(u, a_min=0, a_max=self.resolution[0] - self.pixel_size)
        v = np.clip(v, a_min=0, a_max=self.resolution[1] - self.pixel_size)

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
            uv[0] : uv[0] + self.pixel_size,
            uv[1] : uv[1] + self.pixel_size,
        ] = self.pixel_color

        image = ToTensor()(image)
        target = torch.as_tensor(uv, dtype=torch.float32)

        return image, target


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = CoordinateRegression(DatasetConfig())

    # Visualize one instance.
    image, _ = dataset[np.random.randint(len(dataset))]
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()

    # Plot target distribution.
    targets = []
    for _, target in dataset:
        targets.append(target.numpy())
    targets = np.asarray(targets)

    plt.scatter(targets[:, 0], targets[:, 1], marker="x", c="black")
    plt.xlim(0 - 2, 100 + 2)
    plt.ylim(0 - 2, 100 + 2)
    plt.show()
