import numpy as np

from src.training.kfold import run_kfold_evaluation_only


class IdentityModel:
    def __init__(self) -> None:
        self.forward_calls = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.forward_calls += 1
        return x


def test_kfold_eval_only_uses_forward_and_covers_all_samples(tmp_path) -> None:
    rng = np.random.default_rng(11)
    clean_images = rng.random((11, 4, 4, 1), dtype=np.float32)

    model = IdentityModel()
    output_csv = tmp_path / "kfold_eval.csv"

    result = run_kfold_evaluation_only(
        model=model,
        clean_images=clean_images,
        k_folds=5,
        batch_size=3,
        seed=11,
        noise_type="gaussian",
        noise_std=0.0,
        output_csv_path=output_csv,
    )

    fold_results = result["fold_results"]
    summary = result["summary"]

    assert len(fold_results) == 5
    assert model.forward_calls > 0

    total_val_samples = sum(int(row["val_samples"]) for row in fold_results)
    assert total_val_samples == clean_images.shape[0]

    for row in fold_results:
        assert np.isclose(row["val_mse"], 0.0)
        assert np.isclose(row["val_rmse"], 0.0)
        assert row["val_psnr"] == float("inf")

    assert np.isclose(summary["val_mse_mean"], 0.0)
    assert np.isclose(summary["val_mse_std"], 0.0)
    assert np.isclose(summary["val_rmse_mean"], 0.0)
    assert np.isclose(summary["val_rmse_std"], 0.0)
    assert summary["val_psnr_mean"] == float("inf")

    assert output_csv.exists()
