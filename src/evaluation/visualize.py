from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_denoising_grid(
    clean_images: np.ndarray,
    noisy_images: np.ndarray,
    reconstructed_images: np.ndarray,
    output_path: Path,
    num_images: int = 8,
) -> None:
    count = min(num_images, clean_images.shape[0])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, count, figsize=(2 * count, 6))

    for index in range(count):
        axes[0, index].imshow(noisy_images[index])
        axes[0, index].axis("off")
        axes[0, index].set_title("Noisy")

        axes[1, index].imshow(reconstructed_images[index])
        axes[1, index].axis("off")
        axes[1, index].set_title("Reconstructed")

        axes[2, index].imshow(clean_images[index])
        axes[2, index].axis("off")
        axes[2, index].set_title("Clean")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
