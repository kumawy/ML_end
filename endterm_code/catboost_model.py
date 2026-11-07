"""Native-categorical price models selected without accepting test data.

The six fixed configurations use shuffled training-only CV. A separate final
fit uses the supplied validation split only for its stopping/checkpoint metric.
Every public prediction and every selection score is an original KZT price.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, __version__ as catboost_version
from sklearn.model_selection import KFold

import data


def prepare_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Apply deterministic conversions, retaining native numeric missingness."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("CatBoost requires a raw pandas DataFrame with named features.")
    missing = set(feature_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CatBoost predictor columns: {sorted(missing)}")
    result = frame.loc[:, feature_columns].copy()
    for column in categorical_columns:
        result[column] = result[column].astype(object).where(result[column].notna(), "__missing__").astype(str)
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    return result


def _original_prices(predictions, log_target: bool) -> np.ndarray:
    values = np.asarray(predictions, dtype=float).ravel()
    if log_target:
        max_log = float(np.log(np.finfo(np.float64).max) - 2.0)
        values = np.nan_to_num(values, nan=0.0, posinf=max_log, neginf=0.0)
        return np.expm1(np.clip(values, 0.0, max_log))
    values = np.nan_to_num(values, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    return np.maximum(values, 0.0)


@dataclass
class CatBoostPriceRegressor:
    """Fitted native-feature estimator exposing finite, nonnegative KZT prices."""

    estimator: Any
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    log_target: bool = False

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = prepare_features(
            frame, self.feature_columns, self.numeric_columns, self.categorical_columns
        )
        return _original_prices(self.estimator.predict(prepared), self.log_target)


class LogTargetMAE:
    """CatBoost checkpoint metric: MAE after inverse log1p, in original KZT."""

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        # Keep the metric independent of learned transforms; CatBoost can JIT
        # this scalar loop, and each CV fold needs no target-scaler state.
        error_sum, weight_sum = 0.0, 0.0
        max_log = 707.782712893384
        for index in range(len(target)):
            prediction = approxes[0][index]
            if math.isnan(prediction):
                prediction = 0.0
            prediction = math.expm1(min(max(prediction, 0.0), max_log))
            original_target = math.expm1(target[index])
            observation_weight = 1.0 if weight is None else weight[index]
            error_sum += abs(prediction - original_target) * observation_weight
            weight_sum += observation_weight
        return error_sum, weight_sum

    def get_final_error(self, error, weight):
        return error / weight if weight > 0 else 0.0


def _validate_targets(target, length: int, name: str) -> np.ndarray:
    values = np.asarray(target, dtype=float).ravel()
    if len(values) != length or length == 0:
        raise ValueError(f"{name} must have one target per observation and cannot be empty.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("CatBoost targets must be finite nonnegative KZT prices.")
    return values


def _fit_configuration(train, train_y, validation, validation_y, categorical, depth, log_target, seed):
    model = CatBoostRegressor(
        iterations=1000, depth=int(depth), learning_rate=0.05, l2_leaf_reg=5.0,
        loss_function="RMSE" if log_target else "MAE",
        eval_metric=LogTargetMAE() if log_target else "MAE",
        random_seed=int(seed), thread_count=2, task_type="CPU",
        boosting_type="Plain", grow_policy="SymmetricTree",
        one_hot_max_size=0,
        verbose=False, allow_writing_files=False,
    )
    model.fit(
        train, np.log1p(train_y) if log_target else train_y,
        cat_features=categorical,
        eval_set=(validation, np.log1p(validation_y) if log_target else validation_y),
        early_stopping_rounds=60, use_best_model=True,
    )
    return model


def _save_history(model, log_target: bool, destination: Path) -> tuple[pd.DataFrame, str]:
    evaluations = model.get_evals_result()
    metric_name = "LogTargetMAE" if log_target else "MAE"
    validation_name = next(name for name in evaluations if name != "learn")
    train_mae = evaluations["learn"][metric_name]
    val_mae = evaluations[validation_name][metric_name]
    objective = "RMSE" if log_target else "MAE"
    history = pd.DataFrame({
        "iteration": np.arange(1, len(val_mae) + 1),
        "train_mae_kzt": train_mae,
        "val_mae_kzt": val_mae,
        "train_objective_metric": evaluations["learn"][objective],
        "val_objective_metric": evaluations[validation_name][objective],
    })
    history["is_best_iteration"] = history.iteration.eq(model.get_best_iteration() + 1)
    history.to_csv(destination / "catboost_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    ax.plot(history.iteration, history.train_mae_kzt / 1e6, label="Training", color="#426b9b")
    ax.plot(history.iteration, history.val_mae_kzt / 1e6, label="Validation", color="#18704a")
    best = history.loc[history.is_best_iteration].iloc[0]
    ax.scatter([best.iteration], [best.val_mae_kzt / 1e6], marker="*", s=100, color="#18704a", zorder=3)
    ax.set(xlabel="Boosting iteration", ylabel="MAE (million KZT)", title="Native categorical CatBoost; star marks retained checkpoint")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)
    ax.legend()
    fig.savefig(destination / "catboost_history.png", dpi=180, facecolor="white")
    plt.close(fig)
    return history, metric_name


def train_catboost(
    X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
    output_dir: Path, seed: int = 42,
) -> tuple[CatBoostPriceRegressor, dict]:
    """Choose six fixed candidates by train CV, then checkpoint on validation.

    No test observations or labels are accepted. No rows from external
    validation are fitted. All categorical conversions are deterministic;
    CatBoost learns categorical statistics within each actual fit.
    """
    if not isinstance(X_train, pd.DataFrame) or X_train.columns.has_duplicates:
        raise ValueError("Training predictors must be a DataFrame with unique column names.")
    features = list(X_train.columns)
    # Resolve the enhanced data module's feature lists at call time.
    numeric = [column for column in features if column in data.NUMERIC]
    categorical = [column for column in features if column in data.CATEGORICAL]
    if set(features) != set(numeric + categorical) or not features:
        raise ValueError("All CatBoost predictors must be declared in data.NUMERIC or data.CATEGORICAL.")
    train = prepare_features(X_train, features, numeric, categorical)
    validation = prepare_features(X_val, features, numeric, categorical)
    train_y = _validate_targets(y_train, len(train), "Training split")
    validation_y = _validate_targets(y_val, len(validation), "Validation split")
    if len(train) < 3:
        raise ValueError("Three-fold CatBoost CV requires at least three training observations.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    folds = list(KFold(n_splits=3, shuffle=True, random_state=int(seed)).split(train))
    candidates = []
    for log_target in (False, True):
        for depth in (4, 6, 8):
            candidate_id = f"catboost_{'log_rmse' if log_target else 'raw_mae'}_depth{depth}"
            scores, best_iterations, iterations_run = [], [], []
            for training_indices, validation_indices in folds:
                fitted = _fit_configuration(
                    train.iloc[training_indices], train_y[training_indices],
                    train.iloc[validation_indices], train_y[validation_indices],
                    categorical, depth, log_target, seed,
                )
                predictions = _original_prices(fitted.predict(train.iloc[validation_indices]), log_target)
                scores.append(float(np.mean(np.abs(train_y[validation_indices] - predictions))))
                best_iterations.append(int(fitted.get_best_iteration() + 1))
                evaluation = fitted.get_evals_result()
                validation_key = next(name for name in evaluation if name != "learn")
                metric_key = "LogTargetMAE" if log_target else "MAE"
                iterations_run.append(len(evaluation[validation_key][metric_key]))
            candidates.append({
                "candidate_id": candidate_id, "depth": depth,
                "loss_function": "RMSE" if log_target else "MAE",
                "target_transform": "log1p" if log_target else "identity",
                "mean_cv_mae_kzt": float(np.mean(scores)),
                "std_cv_mae_kzt": float(np.std(scores, ddof=0)),
                "fold_mae_kzt": scores,
                "fold_best_iterations": best_iterations,
                "fold_iterations_run": iterations_run,
            })
    selected = min(candidates, key=lambda candidate: candidate["mean_cv_mae_kzt"])
    rows = []
    for candidate in candidates:
        row = {key: value for key, value in candidate.items() if not isinstance(value, list)}
        for index in range(3):
            row[f"fold{index + 1}_mae_kzt"] = candidate["fold_mae_kzt"][index]
            row[f"fold{index + 1}_best_iteration"] = candidate["fold_best_iterations"][index]
            row[f"fold{index + 1}_iterations_run"] = candidate["fold_iterations_run"][index]
        row["selected_by_cv"] = candidate is selected
        rows.append(row)
    pd.DataFrame(rows).to_csv(destination / "catboost_cv.csv", index=False)
    log_target = selected["target_transform"] == "log1p"
    fitted = _fit_configuration(
        train, train_y, validation, validation_y,
        categorical, selected["depth"], log_target, seed,
    )
    wrapped = CatBoostPriceRegressor(fitted, features, numeric, categorical, log_target)
    history, checkpoint_metric = _save_history(fitted, log_target, destination)
    validation_mae = float(np.mean(np.abs(validation_y - wrapped.predict(X_val))))
    report = {
        "model": "CatBoost", "catboost_version": catboost_version,
        "selected_candidate_id": selected["candidate_id"],
        "best_validation_mae_kzt": validation_mae,
        "selected_cv_mae_kzt": selected["mean_cv_mae_kzt"],
        "selected_cv_std_mae_kzt": selected["std_cv_mae_kzt"],
        "best_iteration": int(fitted.get_best_iteration() + 1),
        "iterations_run": len(history), "retained_tree_count": int(fitted.tree_count_),
        "configuration": {
            "depth": selected["depth"], "loss_function": selected["loss_function"],
            "target_transform": selected["target_transform"],
            "iterations_max": 1000, "learning_rate": 0.05, "l2_leaf_reg": 5.0,
            "thread_count": 2, "random_seed": int(seed), "patience": 60,
            "task_type": "CPU", "boosting_type": "Plain", "grow_policy": "SymmetricTree",
            "one_hot_max_size": 0,
            "checkpoint_metric": checkpoint_metric,
            "use_best_model": True,
        },
        "fitted_parameters": fitted.get_all_params(),
        "architecture": "Additive gradient boosting of symmetric decision trees; native categorical statistics, no one-hot preprocessing.",
        "numeric_features": numeric, "categorical_features": categorical,
        "feature_columns": features,
        "preprocessing": {
            "numeric": "float64; genuine missing numeric values remain NaN and are handled by CatBoost",
            "categorical": "missing values become __missing__; all categories cast to strings; saved training column order retained",
            "learned_transform_fit_split": "each CV training fold; final fit uses training only",
        },
        "target_transform": (
            "log1p(price_KZT); predict max(0, expm1(predicted_log_price)); no target scaler; log clipped only for numerical safety"
            if log_target else "identity price_KZT; negative and nonfinite predictions guarded to return finite nonnegative prices"
        ),
        "cv": {
            "n_splits": 3, "shuffle": True, "random_seed": int(seed),
            "fit_split": "training only; fold validation controls each fit's early stopping",
            "selection_metric": "mean fold MAE in original KZT; standard deviation uses ddof=0",
            "tie_breaking": "first candidate in raw MAE then log RMSE, each depth 4/6/8",
            "candidate_count": len(candidates),
        },
        "final_fit": "training observations only; external validation selects the best MAE-in-KZT checkpoint with patience 60 and at most 1000 iterations",
        "history_metric_precision": "CatBoost evaluates its internally rounded target buffers; inverse-log history metrics can differ by a few KZT from exact original-label errors. Reported CV scores and final validation MAE are recomputed against original KZT targets.",
        "global_model_selection": "Caller compares this final validation MAE with other model families before retrospective test evaluation.",
        "candidates": candidates,
        "artifacts": {
            "cv_csv": "catboost_cv.csv", "history_csv": "catboost_history.csv",
            "history_plot": "catboost_history.png", "report_json": "catboost_report.json",
        },
        "test_data_used": False,
    }
    with (destination / "catboost_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return wrapped, report
