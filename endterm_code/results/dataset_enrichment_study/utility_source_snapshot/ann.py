"""Feedforward price regressor selected using validation MAE in original KZT.

The caller supplies dense predictors produced by a train-fitted preprocessor.
Only the training target fits the standardized log1p target transformation.
No test data is accepted by this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class ANNRegressor:
    """A fitted neural network whose public predictions are prices in KZT."""

    estimator: Any
    target_scaler: StandardScaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        standardized = np.asarray(self.estimator.predict(X), dtype=float).reshape(-1, 1)
        log_prices = self.target_scaler.inverse_transform(standardized).ravel()
        # Guard exponent overflow and negative prices, including extrapolation.
        max_log = float(np.log(np.finfo(np.float64).max) - 2.0)
        log_prices = np.nan_to_num(log_prices, nan=0.0, posinf=max_log, neginf=0.0)
        return np.expm1(np.clip(log_prices, 0.0, max_log))


def _validate_inputs(X_train, y_train, X_val, y_val):
    arrays = tuple(np.asarray(value, dtype=float) for value in (X_train, y_train, X_val, y_val))
    train, train_y, validation, validation_y = arrays
    train_y, validation_y = train_y.ravel(), validation_y.ravel()
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("Training and validation predictors must be 2D with matching feature counts.")
    if len(train) != len(train_y) or len(validation) != len(validation_y):
        raise ValueError("Each predictor matrix must have one target per row.")
    if len(train_y) == 0 or len(validation_y) == 0:
        raise ValueError("Training and validation splits must both contain observations.")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("ANN inputs must be finite after train-fitted preprocessing.")
    if np.any(train_y < 0) or np.any(validation_y < 0):
        raise ValueError("Car prices must be nonnegative KZT values.")
    return train, train_y, validation, validation_y


def _assess_history(rows: list[dict], best_epoch: int, stopped_early: bool) -> dict:
    first, best, final = rows[0], rows[best_epoch - 1], rows[-1]
    recent = rows[-min(20, len(rows)):]
    recent_change = (recent[0]["train_scaled_log_mse"] - final["train_scaled_log_mse"]) / max(
        recent[0]["train_scaled_log_mse"], np.finfo(float).eps
    )
    deterioration = final["val_mae_kzt"] / max(best["val_mae_kzt"], np.finfo(float).eps) - 1.0
    overfitting = (
        deterioration > 0.05
        and final["train_scaled_log_mse"] < best["train_scaled_log_mse"]
    )
    return {
        "stopping_reason": "validation_patience_exhausted" if stopped_early else "maximum_epochs_reached",
        "training_mse_first": first["train_scaled_log_mse"],
        "training_mse_best_checkpoint": best["train_scaled_log_mse"],
        "training_mse_final": final["train_scaled_log_mse"],
        "relative_training_mse_improvement_last_up_to_20_epochs": float(recent_change),
        "validation_mae_deterioration_after_best_fraction": float(deterioration),
        "overfitting_signal": bool(overfitting),
        "assessment": (
            "Training error continues to fall while validation MAE worsens by over 5% after its best epoch; "
            "the saved checkpoint avoids the later deterioration."
            if overfitting else
            "No post-checkpoint overfitting signal under the stated 5% validation-MAE heuristic; "
            "this does not establish generalization outside these splits."
        ),
        "heuristic": "Overfitting signal: final validation MAE > 1.05 * best validation MAE and final train MSE < train MSE at best epoch.",
    }


def _plot_history(history: pd.DataFrame, selected_id: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    palette = ["#18704a", "#426b9b"]
    for color, (candidate_id, frame) in zip(palette, history.groupby("candidate_id", sort=False)):
        label = frame["architecture"].iloc[0]
        axes[0].plot(frame["epoch"], frame["train_scaled_log_mse"], color=color, label=f"{label} train")
        axes[0].plot(frame["epoch"], frame["val_scaled_log_mse"], color=color, linestyle="--", label=f"{label} validation")
        axes[1].plot(frame["epoch"], frame["train_mae_kzt"] / 1e6, color=color, label=f"{label} train")
        axes[1].plot(frame["epoch"], frame["val_mae_kzt"] / 1e6, color=color, linestyle="--", label=f"{label} validation")
        best_row = frame.loc[frame["val_mae_kzt"].idxmin()]
        axes[1].scatter(best_row["epoch"], best_row["val_mae_kzt"] / 1e6, color=color,
                        s=55 if candidate_id == selected_id else 30, zorder=4,
                        marker="*" if candidate_id == selected_id else "o")
    axes[0].set_title("Squared error on standardized log1p(price)")
    axes[0].set_ylabel("MSE")
    axes[1].set_title("Price error; star marks selected checkpoint")
    axes[1].set_ylabel("MAE (million KZT)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def train_ann(X_train, y_train, X_val, y_val, output_dir, seed=42) -> tuple[ANNRegressor, dict]:
    """Fit two fixed architectures, retaining the lowest validation-MAE epoch.

    Adam receives one epoch per ``partial_fit`` call. Validation MAE in KZT
    controls both patience and the selected architecture; held-out test labels
    are neither accepted nor used. The caller controls BLAS thread counts.
    """
    X_train, y_train, X_val, y_val = _validate_inputs(X_train, y_train, X_val, y_val)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler().fit(np.log1p(y_train).reshape(-1, 1))
    scaled_train = scaler.transform(np.log1p(y_train).reshape(-1, 1)).ravel()
    scaled_val = scaler.transform(np.log1p(y_val).reshape(-1, 1)).ravel()
    max_epochs, patience = 400, 40
    config = {
        "activation": "relu", "output_activation": "identity (linear)",
        "solver": "adam", "loss": "squared_error (half squared error plus L2 penalty)",
        "learning_rate_init": 0.001, "alpha": 0.001, "batch_size": 64,
        "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-8,
        "shuffle": True, "max_epochs": max_epochs, "patience": patience,
        "minimum_validation_mae_improvement": 0.0, "random_state": int(seed),
        "early_stopping": "external validation MAE in KZT; sklearn internal early_stopping=False",
    }
    all_rows, candidates = [], []
    selected_model, selected_candidate = None, None
    for architecture in ((64, 32), (128, 64)):
        candidate_id = "ann_" + "_".join(str(width) for width in architecture)
        network = MLPRegressor(
            hidden_layer_sizes=architecture, activation="relu", solver="adam",
            alpha=0.001, batch_size=64, learning_rate_init=0.001,
            random_state=int(seed), shuffle=True, early_stopping=False,
            max_iter=1, tol=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        )
        best_mae, best_epoch, best_network = float("inf"), 0, None
        stale_epochs, rows = 0, []
        for epoch in range(1, max_epochs + 1):
            network.partial_fit(X_train, scaled_train)
            wrapped = ANNRegressor(network, scaler)
            train_scaled_predictions = network.predict(X_train)
            val_scaled_predictions = network.predict(X_val)
            row = {
                "candidate_id": candidate_id, "architecture": "-".join(map(str, architecture)),
                "epoch": epoch,
                "train_scaled_log_mse": float(np.mean((scaled_train - train_scaled_predictions) ** 2)),
                "val_scaled_log_mse": float(np.mean((scaled_val - val_scaled_predictions) ** 2)),
                "train_mae_kzt": float(np.mean(np.abs(y_train - wrapped.predict(X_train)))),
                "val_mae_kzt": float(np.mean(np.abs(y_val - wrapped.predict(X_val)))),
            }
            if not all(np.isfinite(row[key]) for key in (
                "train_scaled_log_mse", "val_scaled_log_mse", "train_mae_kzt", "val_mae_kzt"
            )):
                raise FloatingPointError(f"Non-finite ANN training metric for {candidate_id} at epoch {epoch}.")
            rows.append(row)
            if row["val_mae_kzt"] < best_mae:
                best_mae, best_epoch = row["val_mae_kzt"], epoch
                best_network = deepcopy(network)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break
        for row in rows:
            row["is_best_epoch"] = row["epoch"] == best_epoch
        all_rows.extend(rows)
        checkpoint = rows[best_epoch - 1]
        candidate = {
            "candidate_id": candidate_id, "hidden_layer_sizes": list(architecture),
            "best_epoch": int(best_epoch), "epochs_run": len(rows),
            "best_validation_mae_kzt": best_mae,
            "checkpoint_train_mae_kzt": checkpoint["train_mae_kzt"],
            "checkpoint_train_scaled_log_mse": checkpoint["train_scaled_log_mse"],
            "checkpoint_validation_scaled_log_mse": checkpoint["val_scaled_log_mse"],
            "assessment": _assess_history(rows, best_epoch, stale_epochs >= patience),
        }
        candidates.append(candidate)
        if selected_candidate is None or best_mae < selected_candidate["best_validation_mae_kzt"]:
            selected_model = ANNRegressor(best_network, scaler)
            selected_candidate = candidate
    history = pd.DataFrame(all_rows)
    history.to_csv(destination / "ann_history.csv", index=False)
    _plot_history(history, selected_candidate["candidate_id"], destination / "ann_history.png")
    report = {
        "model": "ANN", "selected_candidate_id": selected_candidate["candidate_id"],
        "hidden_layer_sizes": selected_candidate["hidden_layer_sizes"],
        "best_epoch": selected_candidate["best_epoch"], "epochs_run": selected_candidate["epochs_run"],
        "best_validation_mae_kzt": selected_candidate["best_validation_mae_kzt"],
        "configuration": config, "candidate_architectures": [[64, 32], [128, 64]],
        "target_scaler": {
            "fit_split": "training only", "n_samples_seen": int(scaler.n_samples_seen_),
            "mean_log1p": float(scaler.mean_[0]), "scale_log1p": float(scaler.scale_[0]),
        },
        "target_transform": "z = (log1p(price_KZT) - train_mean) / train_std; prediction = max(0, expm1(z_hat * train_std + train_mean)); log clipped only to avoid numerical overflow.",
        "selection": "Lowest validation MAE in KZT across all candidate epochs; exact ties retain the first checkpoint/candidate.",
        "history_loss": "Ordinary MSE of standardized log1p targets, without the optimizer's one-half factor or L2 term.",
        "assessment": selected_candidate["assessment"], "candidates": candidates,
        "artifacts": {"history_csv": "ann_history.csv", "history_plot": "ann_history.png"},
        "test_data_used": False,
    }
    return selected_model, report
