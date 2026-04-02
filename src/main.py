from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.classification import run_softmax_classification_on_denoised
from src.config import default_config
from src.data import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    load_cifar10,
    normalize_images,
    resize_images,
    select_random_subset,
    validate_cifar10_dataset,
)
from src.evaluation.visualize import save_denoising_grid, save_loss_curve
from src.models import CDAE, FullyConnectedDAE
from src.training import run_kfold_evaluation_only, run_kfold_experiment, train_fold
from src.utils import ExperimentLogger, set_global_seed


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDAE CIFAR-10 denoising with custom NumPy layers")
    parser.add_argument("--mode", choices=["kfold", "single", "kfold_eval"], default="kfold")
    parser.add_argument("--model-type", choices=["conv", "fc"], default="conv")
    parser.add_argument("--fc-hidden-dim", type=int, default=512)
    parser.add_argument("--fc-bottleneck-dim", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--download-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download CIFAR-10 automatically via torchvision on first run.",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--lr-decay-every", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument(
        "--conv-skip-connection-weight",
        type=float,
        default=0.7,
        help="Residual blend weight for conv model output: final=(1-w)*decoded + w*noisy.",
    )
    parser.add_argument(
        "--l1-weight",
        type=float,
        default=0.1,
        help="Optional L1 term weight in reconstruction loss: loss = MSE + l1_weight * MAE",
    )
    parser.add_argument("--noise-type", choices=["gaussian", "salt_pepper"], default="gaussian")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument(
        "--sample-noise-per-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For gaussian noise, resample std per mini-batch from --batch-noise-std-options.",
    )
    parser.add_argument(
        "--batch-noise-std-options",
        type=str,
        default="0.05,0.1",
        help="Comma-separated std values used when --sample-noise-per-batch is enabled.",
    )
    parser.add_argument("--salt-pepper-amount", type=float, default=0.01)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--resize-to", type=int, default=16)
    parser.add_argument(
        "--random-subset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly sample the requested subset size from train/test before training/evaluation.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--resume-checkpoint", type=str, default="")
    parser.add_argument("--save-checkpoint", type=str, default="")
    parser.add_argument("--save-figure", action="store_true")
    parser.add_argument("--figure-name", type=str, default="denoising_preview.png")
    parser.add_argument("--save-loss-curve", action="store_true")
    parser.add_argument("--loss-curve-name", type=str, default="training_loss_curve.png")
    parser.add_argument(
        "--kfold-eval-output",
        type=str,
        default="reports/tables/kfold_eval_summary_latest.csv",
    )
    parser.add_argument("--num-figure-images", type=int, default=8)
    parser.add_argument("--run-classification", action="store_true")
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--classifier-learning-rate", type=float, default=0.1)
    parser.add_argument("--classifier-weight-decay", type=float, default=1e-4)
    parser.add_argument("--classifier-seed", type=int, default=123)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--predictions-output",
        type=str,
        default="reports/tables/test_predictions_sample.csv",
    )
    parser.add_argument("--num-prediction-samples", type=int, default=100)
    parser.add_argument(
        "--track-ssim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute SSIM in addition to MSE/RMSE/PSNR.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def summarize_fold_results(fold_results: list[dict[str, object]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    metric_keys = [
        "val_mse",
        "val_rmse",
        "val_psnr",
        "val_ssim",
        "test_mse",
        "test_rmse",
        "test_psnr",
        "test_ssim",
        "classification_train_accuracy",
        "classification_val_accuracy",
        "classification_test_accuracy",
        "classifier_final_loss",
    ]

    for metric_key in metric_keys:
        values = [float(result[metric_key]) for result in fold_results if metric_key in result]
        if values:
            values_array = np.asarray(values, dtype=np.float64)
            summary[f"{metric_key}_mean"] = float(np.mean(values_array))
            summary[f"{metric_key}_std"] = float(np.std(values_array))

    return summary


def create_noisy_images(
    clean_images: np.ndarray,
    noise_type: str,
    noise_std: float,
    salt_pepper_amount: float,
    seed: int,
) -> np.ndarray:
    if noise_type == "gaussian":
        return add_gaussian_noise(clean_images, std=noise_std, seed=seed)

    return add_salt_pepper_noise(
        clean_images,
        amount=salt_pepper_amount,
        seed=seed,
    )


def parse_batch_noise_std_options(raw_options: str) -> tuple[float, ...]:
    cleaned_options: list[float] = []
    for token in raw_options.split(","):
        token = token.strip()
        if token == "":
            continue
        value = float(token)
        if value <= 0.0:
            raise ValueError("All --batch-noise-std-options values must be > 0.")
        cleaned_options.append(value)

    if not cleaned_options:
        return tuple()

    return tuple(cleaned_options)


def write_prediction_samples(
    output_path: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    max_rows: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = min(max_rows, y_true.shape[0], y_pred.shape[0])

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["sample_index", "true_label", "predicted_label", "correct"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for sample_index in range(row_count):
            true_index = int(y_true[sample_index])
            pred_index = int(y_pred[sample_index])

            writer.writerow(
                {
                    "sample_index": sample_index,
                    "true_label": class_names[true_index],
                    "predicted_label": class_names[pred_index],
                    "correct": int(true_index == pred_index),
                }
            )


def make_log_row(
    *,
    experiment_id: int,
    result: dict[str, object],
    args: argparse.Namespace,
    mode: str,
    fold: int,
    logger: ExperimentLogger,
    checkpoint_path: str,
    train_sample_count: int,
    test_sample_count: int,
    notes: str,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "timestamp_utc": logger.utc_timestamp(),
        "mode": mode,
        "fold": fold,
        "dataset": "CIFAR-10",
        "dataset_train_samples": train_sample_count,
        "dataset_test_samples": test_sample_count,
        "epochs_requested": args.epochs,
        "epochs_completed": result.get("epochs_completed", ""),
        "best_epoch": result.get("best_epoch", ""),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "final_learning_rate": result.get("final_learning_rate", ""),
        "lr_decay": args.lr_decay,
        "lr_decay_every": args.lr_decay_every,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "noise_type": args.noise_type,
        "noise_std": args.noise_std,
        "salt_pepper_amount": args.salt_pepper_amount,
        "model_type": args.model_type,
        "latent_channels": args.latent_channels,
        "conv_skip_connection_weight": args.conv_skip_connection_weight,
        "fc_hidden_dim": args.fc_hidden_dim,
        "fc_bottleneck_dim": args.fc_bottleneck_dim,
        "l1_weight": args.l1_weight,
        "resize_to": args.resize_to,
        "random_subset": int(args.random_subset),
        "max_samples": args.max_samples,
        "max_test_samples": args.max_test_samples,
        "sample_noise_per_batch": int(args.sample_noise_per_batch),
        "batch_noise_std_options": args.batch_noise_std_options,
        "resume_checkpoint": args.resume_checkpoint,
        "checkpoint_path": checkpoint_path,
        "run_classification": int(args.run_classification),
        "classifier_epochs": args.classifier_epochs,
        "classifier_batch_size": args.classifier_batch_size,
        "classifier_learning_rate": args.classifier_learning_rate,
        "classifier_weight_decay": args.classifier_weight_decay,
        "classifier_final_loss": result.get("classifier_final_loss", ""),
        "classification_train_accuracy": result.get("classification_train_accuracy", ""),
        "classification_val_accuracy": result.get("classification_val_accuracy", ""),
        "classification_test_accuracy": result.get("classification_test_accuracy", ""),
        "train_mse": result.get("train_mse", ""),
        "val_mse": result.get("val_mse", ""),
        "test_mse": result.get("test_mse", ""),
        "train_rmse": result.get("train_rmse", ""),
        "val_rmse": result.get("val_rmse", ""),
        "test_rmse": result.get("test_rmse", ""),
        "train_psnr": result.get("train_psnr", ""),
        "val_psnr": result.get("val_psnr", ""),
        "test_psnr": result.get("test_psnr", ""),
        "train_ssim": result.get("train_ssim", ""),
        "val_ssim": result.get("val_ssim", ""),
        "test_ssim": result.get("test_ssim", ""),
        "figure_path": result.get("figure_path", ""),
        "loss_curve_path": result.get("loss_curve_path", ""),
        "predictions_path": result.get("predictions_path", ""),
        "best_fold": result.get("best_fold", ""),
        "is_best_fold": result.get("is_best_fold", ""),
        "best_checkpoint_path": result.get("best_checkpoint_path", ""),
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    cfg = default_config()

    cfg.paths.data_root = Path(args.data_root)
    cfg.train.mode = args.mode
    cfg.train.k_folds = args.k_folds
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.learning_rate
    cfg.noise.noise_type = args.noise_type
    cfg.noise.std = args.noise_std
    cfg.train.max_train_samples = args.max_samples
    cfg.train.random_seed = args.seed

    set_global_seed(cfg.train.random_seed)

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10(
        cfg.paths.data_root,
        download=args.download_dataset,
    )
    dataset_summary = validate_cifar10_dataset(y_train, y_test)
    print(
        "[DATASET CHECK] "
        f"train={dataset_summary['train_samples']} "
        f"test={dataset_summary['test_samples']} "
        f"total={dataset_summary['total_samples']} "
        f"classes={dataset_summary['num_classes']}"
    )
    print(
        "[DATASET CHECK] per_class_total="
        f"{np.asarray(dataset_summary['per_class_total']).tolist()}"
    )
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    if args.resize_to != x_train.shape[1]:
        x_train = resize_images(x_train, target_size=args.resize_to)
        x_test = resize_images(x_test, target_size=args.resize_to)
        print(f"[INFO] Resized inputs to: {x_train.shape[1]}x{x_train.shape[2]}")

    sample_count = min(cfg.train.max_train_samples, x_train.shape[0])
    test_sample_count = min(args.max_test_samples, x_test.shape[0])

    if args.random_subset:
        clean_images, train_labels = select_random_subset(
            x_train,
            y_train,
            max_samples=sample_count,
            seed=cfg.train.random_seed,
        )
        clean_test_images, test_labels = select_random_subset(
            x_test,
            y_test,
            max_samples=test_sample_count,
            seed=cfg.train.random_seed + 1,
        )
        print(
            "[INFO] Using randomized subset: "
            f"train={clean_images.shape[0]} test={clean_test_images.shape[0]}"
        )
    else:
        clean_images = x_train[:sample_count]
        clean_test_images = x_test[:test_sample_count]
        train_labels = y_train[:sample_count]
        test_labels = y_test[:test_sample_count]

    def build_model():
        if args.model_type == "conv":
            model = CDAE(
                input_channels=int(clean_images.shape[-1]),
                latent_channels=args.latent_channels,
                seed=cfg.train.random_seed,
                l1_weight=args.l1_weight,
                skip_connection_weight=args.conv_skip_connection_weight,
            )
        else:
            model = FullyConnectedDAE(
                input_shape=(
                    int(clean_images.shape[1]),
                    int(clean_images.shape[2]),
                    int(clean_images.shape[3]),
                ),
                hidden_dim=args.fc_hidden_dim,
                bottleneck_dim=args.fc_bottleneck_dim,
                seed=cfg.train.random_seed,
                skip_connection_weight=0.2,
                l1_weight=args.l1_weight,
            )

        if args.resume_checkpoint:
            model.load_checkpoint(args.resume_checkpoint)
            print(f"[INFO] Loaded checkpoint for initialization: {args.resume_checkpoint}")

        return model

    logger = ExperimentLogger(cfg.paths.logs_csv_path)
    next_experiment_id = logger.next_experiment_id()
    batch_noise_std_options = parse_batch_noise_std_options(args.batch_noise_std_options)

    if cfg.train.mode == "kfold_eval":
        if not args.resume_checkpoint:
            raise ValueError(
                "kfold_eval mode requires --resume-checkpoint to evaluate a pre-trained model."
            )

        model = build_model()
        eval_result = run_kfold_evaluation_only(
            model=model,
            clean_images=clean_images,
            k_folds=cfg.train.k_folds,
            batch_size=cfg.train.batch_size,
            seed=cfg.train.random_seed,
            noise_type=cfg.noise.noise_type,
            noise_std=cfg.noise.std,
            salt_pepper_amount=args.salt_pepper_amount,
            output_csv_path=args.kfold_eval_output,
            track_ssim=args.track_ssim,
        )

        fold_results = [
            result
            for result in eval_result.get("fold_results", [])
            if isinstance(result, dict)
        ]
        for result in fold_results:
            logger.log_run(
                make_log_row(
                    experiment_id=next_experiment_id,
                    result=result,
                    args=args,
                    mode="kfold_eval",
                    fold=int(result["fold"]),
                    logger=logger,
                    checkpoint_path=args.resume_checkpoint,
                    train_sample_count=sample_count,
                    test_sample_count=test_sample_count,
                    notes="kfold evaluation only (no retraining)",
                )
            )
            next_experiment_id += 1

        eval_summary = eval_result.get("summary", {})
        print("\nK-Fold Evaluation Summary (mean +/- std):")
        print(
            f"  val_mse: {float(eval_summary.get('val_mse_mean', float('nan'))):.8f} "
            f"+/- {float(eval_summary.get('val_mse_std', float('nan'))):.8f}"
        )
        print(
            f"  val_rmse: {float(eval_summary.get('val_rmse_mean', float('nan'))):.8f} "
            f"+/- {float(eval_summary.get('val_rmse_std', float('nan'))):.8f}"
        )
        print(
            f"  val_psnr: {float(eval_summary.get('val_psnr_mean', float('nan'))):.6f} "
            f"+/- {float(eval_summary.get('val_psnr_std', float('nan'))):.6f}"
        )
        if args.track_ssim:
            print(
                f"  val_ssim: {float(eval_summary.get('val_ssim_mean', float('nan'))):.6f} "
                f"+/- {float(eval_summary.get('val_ssim_std', float('nan'))):.6f}"
            )
        return

    noisy_images = create_noisy_images(
        clean_images,
        noise_type=cfg.noise.noise_type,
        noise_std=cfg.noise.std,
        salt_pepper_amount=args.salt_pepper_amount,
        seed=cfg.train.random_seed,
    )
    noisy_test_images = create_noisy_images(
        clean_test_images,
        noise_type=cfg.noise.noise_type,
        noise_std=cfg.noise.std,
        salt_pepper_amount=args.salt_pepper_amount,
        seed=cfg.train.random_seed + 1000,
    )

    if cfg.train.mode == "kfold":
        checkpoint_prefix = ""
        if args.save_checkpoint:
            checkpoint_target = Path(args.save_checkpoint)
            checkpoint_prefix = str(checkpoint_target.with_suffix(""))

        fold_results = run_kfold_experiment(
            clean_images=clean_images,
            noisy_images=noisy_images,
            model_builder=build_model,
            k_folds=cfg.train.k_folds,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            learning_rate=cfg.train.learning_rate,
            seed=cfg.train.random_seed,
            weight_decay=args.weight_decay,
            lr_decay=args.lr_decay,
            lr_decay_every=args.lr_decay_every,
            early_stopping_patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            clean_test=clean_test_images,
            noisy_test=noisy_test_images,
            checkpoint_prefix=checkpoint_prefix or None,
            train_labels=train_labels,
            test_labels=test_labels,
            run_classification=args.run_classification,
            classifier_epochs=args.classifier_epochs,
            classifier_batch_size=args.classifier_batch_size,
            classifier_learning_rate=args.classifier_learning_rate,
            classifier_weight_decay=args.classifier_weight_decay,
            classifier_seed=args.classifier_seed,
            save_figures=args.save_figure,
            figure_output_path=cfg.paths.figure_output_dir / args.figure_name,
            num_figure_images=args.num_figure_images,
            save_predictions=args.save_predictions,
            predictions_output_path=args.predictions_output,
            num_prediction_samples=args.num_prediction_samples,
            class_names=CIFAR10_CLASS_NAMES,
            save_loss_curves=args.save_loss_curve,
            loss_curve_output_path=cfg.paths.figure_output_dir / args.loss_curve_name,
            noise_type=cfg.noise.noise_type,
            salt_pepper_amount=args.salt_pepper_amount,
            batch_noise_std_options=batch_noise_std_options,
            sample_noise_per_batch=args.sample_noise_per_batch,
            track_ssim=args.track_ssim,
        )

        for result in fold_results:
            logger.log_run(
                make_log_row(
                    experiment_id=next_experiment_id,
                    result=result,
                    args=args,
                    mode="kfold",
                    fold=int(result["fold"]),
                    logger=logger,
                    checkpoint_path=str(result.get("checkpoint_path", "")),
                    train_sample_count=sample_count,
                    test_sample_count=test_sample_count,
                    notes=f"custom numpy {args.model_type} denoiser kfold run",
                )
            )
            next_experiment_id += 1

        summary = summarize_fold_results(fold_results)
        print("\nK-Fold Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value:.6f}")

        if args.save_figure:
            figure_base = cfg.paths.figure_output_dir / args.figure_name
            print(
                "[INFO] Saved K-fold denoising figures with suffixes to: "
                f"{figure_base.parent} (e.g., {figure_base.stem}_fold1{figure_base.suffix or '.png'})"
            )

        if args.save_predictions and args.run_classification:
            predictions_base = Path(args.predictions_output)
            print(
                "[INFO] Saved K-fold prediction samples with suffixes to: "
                f"{predictions_base.parent} "
                f"(e.g., {predictions_base.stem}_fold1{predictions_base.suffix or '.csv'})"
            )

        if args.save_loss_curve:
            loss_curve_base = cfg.paths.figure_output_dir / args.loss_curve_name
            print(
                "[INFO] Saved K-fold loss curves with suffixes to: "
                f"{loss_curve_base.parent} "
                f"(e.g., {loss_curve_base.stem}_fold1{loss_curve_base.suffix or '.png'})"
            )

        best_checkpoint_candidates = {
            str(result.get("best_checkpoint_path", ""))
            for result in fold_results
            if str(result.get("best_checkpoint_path", ""))
        }
        if best_checkpoint_candidates:
            best_checkpoint_path = sorted(best_checkpoint_candidates)[0]
            print(f"[INFO] Saved best K-fold checkpoint to: {best_checkpoint_path}")

    else:
        indices = np.arange(clean_images.shape[0])
        train_indices, val_indices = train_test_split(
            indices,
            test_size=args.val_ratio,
            random_state=cfg.train.random_seed,
            shuffle=True,
        )

        noisy_train = noisy_images[train_indices]
        noisy_val = noisy_images[val_indices]
        clean_train = clean_images[train_indices]
        clean_val = clean_images[val_indices]
        train_split_labels = train_labels[train_indices]
        val_split_labels = train_labels[val_indices]

        model = build_model()
        result = train_fold(
            model=model,
            noisy_train=noisy_train,
            clean_train=clean_train,
            noisy_val=noisy_val,
            clean_val=clean_val,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            learning_rate=cfg.train.learning_rate,
            seed=cfg.train.random_seed,
            weight_decay=args.weight_decay,
            lr_decay=args.lr_decay,
            lr_decay_every=args.lr_decay_every,
            early_stopping_patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            noisy_test=noisy_test_images,
            clean_test=clean_test_images,
            noise_type=cfg.noise.noise_type,
            salt_pepper_amount=args.salt_pepper_amount,
            batch_noise_std_options=batch_noise_std_options,
            sample_noise_per_batch=args.sample_noise_per_batch,
            track_ssim=args.track_ssim,
        )

        classification_result: dict[str, object] = {}
        if args.run_classification:
            classification_result = run_softmax_classification_on_denoised(
                denoiser_model=model,
                noisy_train=noisy_train,
                y_train=train_split_labels,
                noisy_val=noisy_val,
                y_val=val_split_labels,
                noisy_test=noisy_test_images,
                y_test=test_labels,
                epochs=args.classifier_epochs,
                batch_size=args.classifier_batch_size,
                learning_rate=args.classifier_learning_rate,
                weight_decay=args.classifier_weight_decay,
                seed=args.classifier_seed,
                denoise_batch_size=args.batch_size,
                verbose=True,
            )

            result.update(
                {
                    key: value
                    for key, value in classification_result.items()
                    if key != "test_predictions"
                }
            )

            if args.save_predictions and "test_predictions" in classification_result:
                write_prediction_samples(
                    output_path=args.predictions_output,
                    y_true=test_labels,
                    y_pred=np.asarray(classification_result["test_predictions"], dtype=np.int64),
                    class_names=CIFAR10_CLASS_NAMES,
                    max_rows=args.num_prediction_samples,
                )
                print(f"[INFO] Saved prediction samples to: {args.predictions_output}")

        checkpoint_path = ""
        if args.save_checkpoint:
            model.save_checkpoint(args.save_checkpoint)
            checkpoint_path = args.save_checkpoint
            print(f"[INFO] Saved checkpoint to: {checkpoint_path}")

        if args.save_figure:
            preview_count = min(args.num_figure_images, noisy_val.shape[0])
            preview_noisy = noisy_val[:preview_count]
            preview_clean = clean_val[:preview_count]
            reconstructed_val = model.forward(preview_noisy)
            figure_path = cfg.paths.figure_output_dir / args.figure_name
            save_denoising_grid(
                clean_images=preview_clean,
                noisy_images=preview_noisy,
                reconstructed_images=reconstructed_val,
                output_path=figure_path,
                num_images=preview_count,
            )
            print(f"[INFO] Saved denoising preview figure to: {figure_path}")

        if args.save_loss_curve:
            loss_curve_path = cfg.paths.figure_output_dir / args.loss_curve_name
            save_loss_curve(
                history=[entry for entry in result.get("history", []) if isinstance(entry, dict)],
                output_path=loss_curve_path,
            )
            result["loss_curve_path"] = str(loss_curve_path)
            print(f"[INFO] Saved training loss curve to: {loss_curve_path}")

        logger.log_run(
            make_log_row(
                experiment_id=next_experiment_id,
                result=result,
                args=args,
                mode="single",
                fold=1,
                logger=logger,
                checkpoint_path=checkpoint_path,
                train_sample_count=sample_count,
                test_sample_count=test_sample_count,
                notes=f"custom numpy {args.model_type} denoiser single run",
            )
        )
        next_experiment_id += 1

        print("\nSingle Split Summary:")
        print(f"  best_epoch: {int(result['best_epoch'])}")
        print(f"  train_mse: {result['train_mse']:.6f}")
        print(f"  val_mse: {result['val_mse']:.6f}")
        if "test_mse" in result:
            print(f"  test_mse: {result['test_mse']:.6f}")
        print(f"  train_psnr: {result['train_psnr']:.6f}")
        print(f"  val_psnr: {result['val_psnr']:.6f}")
        if "test_psnr" in result:
            print(f"  test_psnr: {result['test_psnr']:.6f}")
        if args.track_ssim:
            print(f"  train_ssim: {float(result.get('train_ssim', float('nan'))):.6f}")
            print(f"  val_ssim: {float(result.get('val_ssim', float('nan'))):.6f}")
            if "test_ssim" in result:
                print(f"  test_ssim: {float(result.get('test_ssim', float('nan'))):.6f}")

        if args.run_classification:
            print("\nClassification Summary:")
            print(f"  train_accuracy: {result['classification_train_accuracy']:.4f}")
            print(f"  val_accuracy: {result['classification_val_accuracy']:.4f}")
            if "classification_test_accuracy" in result:
                print(f"  test_accuracy: {result['classification_test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
