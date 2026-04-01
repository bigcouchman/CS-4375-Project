import numpy as np
import pytest

from src.data.cifar10_loader import validate_cifar10_dataset


def test_validate_cifar10_dataset_passes_for_expected_distribution() -> None:
    y_train = np.repeat(np.arange(10, dtype=np.int64), 5000)
    y_test = np.repeat(np.arange(10, dtype=np.int64), 1000)

    summary = validate_cifar10_dataset(y_train, y_test)

    assert summary["train_samples"] == 50000
    assert summary["test_samples"] == 10000
    assert summary["num_classes"] == 10
    assert np.all(np.asarray(summary["per_class_total"]) == 6000)


def test_validate_cifar10_dataset_fails_for_wrong_class_balance() -> None:
    y_train = np.repeat(np.arange(10, dtype=np.int64), 5000)
    y_test = np.repeat(np.arange(10, dtype=np.int64), 1000)
    y_test[0] = 1

    with pytest.raises(ValueError):
        validate_cifar10_dataset(y_train, y_test)
