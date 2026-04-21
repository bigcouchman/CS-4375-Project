from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    project_root: Path = Path(".")
    data_root: Path = Path("data")
    logs_csv_path: Path = Path("experiments/logs/experiment_runs.csv")
    figure_output_dir: Path = Path("reports/figures")


@dataclass
class NoiseConfig:
    noise_type: str = "gaussian"
    std: float = 0.10
    salt_pepper_amount: float = 0.01


@dataclass
class ModelConfig:
    input_height: int = 32
    input_width: int = 32
    input_channels: int = 3
    latent_channels: int = 64


@dataclass
class TrainConfig:
    mode: str = "kfold"
    k_folds: int = 5
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    random_seed: int = 42
    max_train_samples: int = 50000


@dataclass
class ProjectConfig:
    team_members: list[str] = field(
        default_factory=lambda: [
            "Casey Nguyen (CXN220034)",
            "Nguyen Phuc Do (NPD220001)",
        ]
    )
    dataset_name: str = "CIFAR-10"
    paths: PathsConfig = field(default_factory=PathsConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def default_config() -> ProjectConfig:
    return ProjectConfig()
