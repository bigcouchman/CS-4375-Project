from __future__ import annotations

import pickle
import tarfile
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    from torchvision import datasets as tv_datasets

    HAS_TORCHVISION = True
except Exception:  # pragma: no cover - fallback path is covered separately
    tv_datasets = None
    HAS_TORCHVISION = False

try:
    from numpy.exceptions import VisibleDeprecationWarning
except ImportError:  # pragma: no cover - fallback for older NumPy versions
    VisibleDeprecationWarning = Warning

CIFAR10_NUM_CLASSES = 10
CIFAR10_TRAIN_SAMPLES = 50_000
CIFAR10_TEST_SAMPLES = 10_000
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_NAME = "cifar-10-python.tar.gz"
CIFAR10_EXTRACTED_DIR = "cifar-10-batches-py"

TRAIN_BATCH_FILES = (
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
)
TEST_BATCH_FILE = "test_batch"


def _read_pickle(file_path: Path) -> dict[str, Any]:
    with file_path.open("rb") as file:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                category=VisibleDeprecationWarning,
            )
            return pickle.load(file, encoding="latin1")


def _get_batch_value(batch_dict: dict[str, Any], key: str) -> Any:
    if key in batch_dict:
        return batch_dict[key]

    key_bytes = key.encode("utf-8")
    if key_bytes in batch_dict:
        return batch_dict[key_bytes]

    raise KeyError(f"Missing key '{key}' in CIFAR-10 batch dictionary.")


def _reshape_images(flat_images: np.ndarray) -> np.ndarray:
    images_nchw = flat_images.reshape(-1, 3, 32, 32)
    return images_nchw.transpose(0, 2, 3, 1)


def _required_batch_paths(extracted_dir: Path) -> list[Path]:
    return [extracted_dir / name for name in TRAIN_BATCH_FILES + (TEST_BATCH_FILE,)]


def _download_and_extract_official_python_version(dataset_root: Path) -> Path:
    archive_path = dataset_root / CIFAR10_ARCHIVE_NAME
    extracted_dir = dataset_root / CIFAR10_EXTRACTED_DIR

    if all(path.exists() for path in _required_batch_paths(extracted_dir)):
        return extracted_dir

    if not archive_path.exists():
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(path=dataset_root)

    # Archive is not required after extraction, so remove it to save storage.
    if archive_path.exists():
        archive_path.unlink()

    missing = [str(path) for path in _required_batch_paths(extracted_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Official CIFAR-10 python version extraction failed. "
            f"Missing paths: {missing}"
        )

    return extracted_dir


def _load_cifar10_from_python_batches(extracted_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []

    for batch_name in TRAIN_BATCH_FILES:
        batch_dict = _read_pickle(extracted_dir / batch_name)
        train_images.append(np.asarray(_get_batch_value(batch_dict, "data")))
        train_labels.append(np.asarray(_get_batch_value(batch_dict, "labels"), dtype=np.int64))

    train_flat = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    test_dict = _read_pickle(extracted_dir / TEST_BATCH_FILE)
    test_flat = np.asarray(_get_batch_value(test_dict, "data"))
    y_test = np.asarray(_get_batch_value(test_dict, "labels"), dtype=np.int64)

    x_train = _reshape_images(train_flat)
    x_test = _reshape_images(test_flat)

    return x_train, y_train, x_test, y_test


def load_cifar10(
    dataset_root: str | Path,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 using official torchvision dataset utility.

    Preferred path:
      torchvision.datasets.CIFAR10

    Fallback path (when torchvision is unavailable):
      official CIFAR-10 python-version archive from
      https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    """
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    if HAS_TORCHVISION:
        train_data = tv_datasets.CIFAR10(root=str(dataset_root), train=True, download=download)
        test_data = tv_datasets.CIFAR10(root=str(dataset_root), train=False, download=download)

        x_train = np.asarray(train_data.data)
        y_train = np.asarray(train_data.targets, dtype=np.int64)
        x_test = np.asarray(test_data.data)
        y_test = np.asarray(test_data.targets, dtype=np.int64)
        return x_train, y_train, x_test, y_test

    extracted_dir = dataset_root / CIFAR10_EXTRACTED_DIR
    has_extracted = all(path.exists() for path in _required_batch_paths(extracted_dir))

    if not has_extracted:
        if not download:
            raise FileNotFoundError(
                "CIFAR-10 not found and download is disabled. "
                f"Expected extracted data under: {extracted_dir}"
            )
        extracted_dir = _download_and_extract_official_python_version(dataset_root)

    return _load_cifar10_from_python_batches(extracted_dir)


def get_cifar10_dataset_summary(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    per_class_train = np.bincount(y_train, minlength=CIFAR10_NUM_CLASSES)
    per_class_test = np.bincount(y_test, minlength=CIFAR10_NUM_CLASSES)
    per_class_total = per_class_train + per_class_test

    return {
        "train_samples": int(y_train.shape[0]),
        "test_samples": int(y_test.shape[0]),
        "total_samples": int(y_train.shape[0] + y_test.shape[0]),
        "num_classes": int(max(np.max(y_train), np.max(y_test)) + 1),
        "per_class_train": per_class_train,
        "per_class_test": per_class_test,
        "per_class_total": per_class_total,
    }


def validate_cifar10_dataset(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    summary = get_cifar10_dataset_summary(y_train, y_test)

    expected_train_per_class = CIFAR10_TRAIN_SAMPLES // CIFAR10_NUM_CLASSES
    expected_test_per_class = CIFAR10_TEST_SAMPLES // CIFAR10_NUM_CLASSES

    if summary["train_samples"] != CIFAR10_TRAIN_SAMPLES:
        raise ValueError(
            f"Unexpected train sample count: {summary['train_samples']} != {CIFAR10_TRAIN_SAMPLES}"
        )

    if summary["test_samples"] != CIFAR10_TEST_SAMPLES:
        raise ValueError(
            f"Unexpected test sample count: {summary['test_samples']} != {CIFAR10_TEST_SAMPLES}"
        )

    if summary["num_classes"] != CIFAR10_NUM_CLASSES:
        raise ValueError(
            f"Unexpected class count: {summary['num_classes']} != {CIFAR10_NUM_CLASSES}"
        )

    per_class_train = np.asarray(summary["per_class_train"])
    per_class_test = np.asarray(summary["per_class_test"])
    per_class_total = np.asarray(summary["per_class_total"])

    if not np.all(per_class_train == expected_train_per_class):
        raise ValueError(
            "Train split class balance mismatch. "
            f"Expected each class to have {expected_train_per_class} samples, got {per_class_train.tolist()}."
        )

    if not np.all(per_class_test == expected_test_per_class):
        raise ValueError(
            "Test split class balance mismatch. "
            f"Expected each class to have {expected_test_per_class} samples, got {per_class_test.tolist()}."
        )

    if not np.all(per_class_total == (expected_train_per_class + expected_test_per_class)):
        raise ValueError(
            "Overall class balance mismatch. "
            f"Expected each class to have 6000 samples, got {per_class_total.tolist()}."
        )

    return summary
