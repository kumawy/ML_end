"""Fixed, validation-checkpointed ANN pools; no test data is accepted.

This is the corrected-plus-literal-steering scratch protocol made reproducible:
five fixed seeds, one architecture, two checkpoint criteria, and all-member
equal means. Validation controls checkpoints and the small predeclared blend
grid. It is not an independent estimate of generalization.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from ann import ANNRegressor
import data
from ensemble import Component, PredictionEnsemble
from structured_features import add_structured_features

ROBUST_NUMERIC = tuple(data.CORRECTED_NUMERIC)
ROBUST_CATEGORICAL = (*data.CORRECTED_CATEGORICAL, 'steering')
ROBUST_FEATURES = (*ROBUST_NUMERIC, *ROBUST_CATEGORICAL)
_ALLOWED_ADDITIONAL_NUMERIC = ('log_vehicle_age', 'log_annual_mileage_proxy')
_POOL_NAMES = {'mae_kzt': 'ANN pool MAE', 'dual_score': 'ANN pool dual'}


@dataclass(frozen=True)
class _PoolConfig:
    """Private bounded override for fast behavioral tests; product uses defaults."""

    seeds: tuple = (11, 23, 42, 71, 101)
    hidden_layer_sizes: tuple = (128, 64)
    activation: str = 'relu'
    solver: str = 'adam'
    alpha: float = .001
    batch_size: int = 64
    learning_rate_init: float = .001
    max_epochs: int = 300
    patience: int = 45
    min_frequency: int = 2

    def __post_init__(self):
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or not self.hidden_layer_sizes or min(self.hidden_layer_sizes) < 1
                or self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1
                or self.min_frequency < 1 or self.alpha < 0 or self.learning_rate_init <= 0):
            raise ValueError('ANN pool configuration must have positive sizes and distinct seeds')


DEFAULT_CONFIG = _PoolConfig()


def _additional_numeric_inputs(additional_numeric):
    """Accept only distinct, explicitly declared age/usage proxy columns."""
    if isinstance(additional_numeric, str):
        raise ValueError('additional_numeric must contain distinct declared age/usage input names')
    try:
        names = tuple(additional_numeric)
    except TypeError as exc:
        raise ValueError('additional_numeric must contain distinct declared age/usage input names') from exc
    if (any(not isinstance(name, str) or name not in _ALLOWED_ADDITIONAL_NUMERIC for name in names)
            or len(names) != len(set(names))):
        raise ValueError('additional_numeric permits only distinct log_vehicle_age and log_annual_mileage_proxy inputs')
    return names


def make_robust_preprocessor(min_frequency=2, *, additional_numeric=()):
    """Fresh transformer, with optional declared age/usage columns.

    Fit only on training rows. The default contract is unchanged. Optional
    proxy inputs join the existing median-imputation/scaling pipeline, without
    introducing a separate transformer or extra missingness indicators.
    """
    numeric_inputs = (*ROBUST_NUMERIC, *_additional_numeric_inputs(additional_numeric))
    return ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), list(numeric_inputs)),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                     min_frequency=min_frequency, sparse_output=False)),
        ]), list(ROBUST_CATEGORICAL)),
    ], sparse_threshold=0)


def _target(values, name):
    result = np.asarray(values, dtype=float).ravel()
    if not len(result) or not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError(f'{name} targets must be finite, positive and nonempty')
    return result


def _prediction(values, target, name):
    result = np.asarray(values, dtype=float).ravel()
    if result.shape != target.shape or not np.isfinite(result).all():
        raise ValueError(f'{name} predictions must be finite and align with validation targets')
    return result


def _raw_metrics(target, prediction):
    relative = np.abs(target - prediction) / target
    return {
        'mae_kzt': float(np.mean(np.abs(target - prediction))),
        'within_20pct': float(np.mean(relative <= .20)),
        'median_ape': float(np.median(relative)),
    }


def _metrics(target, prediction, baseline):
    result = _raw_metrics(target, prediction)
    # Identical arithmetic and ordering to the frozen scratch checkpoint score.
    result['dual_score'] = .5 * (result['mae_kzt'] / baseline['mae_kzt']
                                + (1 - result['within_20pct']) / (1 - baseline['within_20pct']))
    return result


def _reference(target, prediction):
    result = _raw_metrics(target, prediction)
    if result['mae_kzt'] <= 0 or result['within_20pct'] >= 1:
        raise ValueError('The normalized dual score requires positive baseline MAE and miss fraction')
    return result


def _plot_history(history, members, destination):
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for seed, rows in history.groupby('seed', sort=False):
        line, = axes[0].plot(rows.epoch, rows.val_scaled_log_mse, label=f'{seed}: validation', alpha=.8)
        axes[0].plot(rows.epoch, rows.train_scaled_log_mse, '--', color=line.get_color(), alpha=.65)
        axes[1].plot(rows.epoch, rows.val_mae_kzt / 1e6, color=line.get_color(), label=str(seed))
        axes[2].plot(rows.epoch, rows.val_within_20pct * 100, color=line.get_color())
        member = next(item for item in members if item['seed'] == int(seed))
        for metric, marker in [('mae_kzt', 'o'), ('dual_score', 'x')]:
            selected_epoch = member['checkpoints'][metric]['best_epoch']
            selected = rows.loc[rows.epoch.eq(selected_epoch)].iloc[0]
            axes[1].scatter(selected_epoch, selected.val_mae_kzt / 1e6, marker=marker,
                            color=line.get_color(), zorder=3)
    axes[0].set_ylabel('Standardized log MSE')
    axes[0].set_title('ANN pool convergence: solid validation, dashed training')
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_ylabel('Validation MAE, million KZT')
    axes[1].set_title('Best MAE checkpoint: circle; best dual checkpoint: cross')
    axes[2].set_ylabel('Validation within ±20%, %')
    axes[2].set_xlabel('Epoch')
    for axis in axes:
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def fit_robust_pool(train_frame, y_train, validation_frame, y_val,
                    baseline_predictions, output_dir, seed=42, *, additional_numeric=(), _config=None):
    """Fit two fixed ANN pools using training and validation only.

    The seed argument records the caller's split seed; network seeds are the
    fixed five-member protocol. Input frames are copied before literal steering
    extraction. Returned ``PredictionEnsemble`` objects require
    ``{'enhanced': add_structured_features(frame)}`` at prediction time.
    Optional ``additional_numeric`` columns must already be present, derived
    with ``dataset_features.add_dataset_features``. They are allowlisted and
    local to this call; module feature constants and default pool names remain
    unchanged.
    """
    additional_numeric = _additional_numeric_inputs(additional_numeric)
    numeric_inputs = (*ROBUST_NUMERIC, *additional_numeric)
    features = (*numeric_inputs, *ROBUST_CATEGORICAL)
    feature_set = 'corrected_plus_literal_steering'
    if set(additional_numeric) == set(_ALLOWED_ADDITIONAL_NUMERIC):
        feature_set = 'corrected_steering_plus_age_usage'
    elif additional_numeric:
        feature_set += '_plus_' + '_and_'.join(additional_numeric)
    config = DEFAULT_CONFIG if _config is None else _config
    if not isinstance(config, _PoolConfig):
        raise TypeError('_config must be a _PoolConfig')
    training_target, validation_target = _target(y_train, 'Training'), _target(y_val, 'Validation')
    if len(train_frame) != len(training_target) or len(validation_frame) != len(validation_target):
        raise ValueError('Training and validation frames must align with their targets')
    baseline_prediction = _prediction(baseline_predictions, validation_target, 'Baseline')
    baseline = _reference(validation_target, baseline_prediction)
    training = add_structured_features(train_frame)
    validation = add_structured_features(validation_frame)
    missing = set(features) - set(training.columns)
    missing |= set(features) - set(validation.columns)
    if missing:
        raise ValueError(f'Missing robust ANN predictors: {sorted(missing)}')
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / 'robust_ann_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    prep = make_robust_preprocessor(config.min_frequency, additional_numeric=additional_numeric)
    x = prep.fit_transform(training)
    v = prep.transform(validation)
    if not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError('Preprocessed ANN inputs must be finite')
    scaler = StandardScaler().fit(np.log1p(training_target).reshape(-1, 1))
    z = scaler.transform(np.log1p(training_target).reshape(-1, 1)).ravel()
    validation_z = scaler.transform(np.log1p(validation_target).reshape(-1, 1)).ravel()
    histories, members = [], []
    pools = {key: [] for key in _POOL_NAMES}
    with threadpool_limits(limits=1):
        for network_seed in config.seeds:
            network = MLPRegressor(
                hidden_layer_sizes=config.hidden_layer_sizes, activation=config.activation,
                solver=config.solver, alpha=config.alpha, batch_size=config.batch_size,
                learning_rate_init=config.learning_rate_init, random_state=network_seed,
                shuffle=True, early_stopping=False, max_iter=1, tol=0,
            )
            best = {key: (float('inf'), 0, None) for key in pools}
            for epoch in range(1, config.max_epochs + 1):
                network.partial_fit(x, z)
                wrapped = ANNRegressor(network, scaler)
                values = _metrics(validation_target, wrapped.predict(v), baseline)
                row = {
                    'feature_set': feature_set, 'seed': int(network_seed),
                    'epoch': epoch,
                    'train_scaled_log_mse': float(np.mean((z - network.predict(x)) ** 2)),
                    'val_scaled_log_mse': float(np.mean((validation_z - network.predict(v)) ** 2)),
                    'train_mae_kzt': float(np.mean(np.abs(training_target - wrapped.predict(x)))),
                    'val_mae_kzt': values['mae_kzt'], 'val_within_20pct': values['within_20pct'],
                    'mae_kzt': values['mae_kzt'], 'within_20pct': values['within_20pct'],
                    'median_ape': values['median_ape'], 'dual_score': values['dual_score'],
                    'best_mae': values['mae_kzt'] < best['mae_kzt'][0],
                    'best_dual': values['dual_score'] < best['dual_score'][0],
                }
                histories.append(row)
                for key in pools:
                    if values[key] < best[key][0]:
                        best[key] = (values[key], epoch, deepcopy(network))
                # Stop only after neither checkpoint has improved for patience
                # epochs. Exact ties retain the first checkpoint, as in scratch.
                if epoch - max(item[1] for item in best.values()) >= config.patience:
                    break
            checkpoints = {}
            seed_history = [row for row in histories if row['seed'] == int(network_seed)]
            for key in pools:
                best_value, best_epoch, best_network = best[key]
                wrapped = ANNRegressor(best_network, scaler)
                component = Component(wrapped, 'enhanced', prep)
                pools[key].append(component)
                values = _metrics(validation_target, wrapped.predict(v), baseline)
                checkpoint_path = Path('robust_ann_checkpoints') / f'seed_{network_seed}_{key}.joblib'
                joblib.dump(component, destination / checkpoint_path)
                selected_row = next(row for row in seed_history if row['epoch'] == best_epoch)
                checkpoints[key] = {
                    'best_epoch': best_epoch, 'best_value': float(best_value),
                    'validation_mae_kzt': values['mae_kzt'],
                    'validation_within_20pct': values['within_20pct'], 'dual_score': values['dual_score'],
                    'train_scaled_log_mse': selected_row['train_scaled_log_mse'],
                    'val_scaled_log_mse': selected_row['val_scaled_log_mse'],
                    'generalization_gap_scaled_log_mse': selected_row['val_scaled_log_mse'] - selected_row['train_scaled_log_mse'],
                    'validation_log_mse_change_after_checkpoint': seed_history[-1]['val_scaled_log_mse'] - selected_row['val_scaled_log_mse'],
                    'validation_mae_change_after_checkpoint_kzt': seed_history[-1]['val_mae_kzt'] - selected_row['val_mae_kzt'],
                    'artifact': checkpoint_path.as_posix(),
                }
            members.append({
                'seed': int(network_seed), 'epochs_run': epoch,
                'stopped_after_both_checkpoints_stale': bool(epoch - max(item[1] for item in best.values()) >= config.patience),
                'checkpoints': checkpoints,
                'assessment': 'Checkpoint losses and post-checkpoint deltas describe validation behavior; the validation split also selects checkpoints, so the gap is descriptive rather than independent overfitting evidence.',
            })
        models, candidate_metrics, model_files = {}, {}, {}
        for key, components in pools.items():
            name = _POOL_NAMES[key]
            pool = PredictionEnsemble(components, [1. / len(components)] * len(components))
            models[name] = pool
            values = _metrics(validation_target, pool.predict({'enhanced': validation}), baseline)
            candidate_metrics[name] = {**values, 'validation_mae_kzt': values['mae_kzt'],
                                       'validation_within_20pct': values['within_20pct'],
                                       'checkpoint_metric': key}
            filename = f'robust_ann_pool_{"mae" if key == "mae_kzt" else "dual"}.joblib'
            joblib.dump(pool, destination / filename)
            model_files[name] = filename
    history = pd.DataFrame(histories)
    history.to_csv(destination / 'robust_ann_history.csv', index=False)
    _plot_history(history, members, destination / 'robust_ann_history.png')
    report = {
        'model': 'Fixed five-seed ANN pools', 'configuration': asdict(config), 'split_seed': int(seed),
        'representation': 'enhanced', 'feature_set': feature_set,
        'numeric_inputs': list(numeric_inputs), 'categorical_inputs': list(ROBUST_CATEGORICAL),
        'additional_numeric_inputs': list(additional_numeric),
        'input_features': int(x.shape[1]), 'training_rows': len(training), 'validation_rows': len(validation),
        'preprocessing_fit_split': 'training only',
        'target_scaler': {'fit_split': 'training only', 'n_samples_seen': int(scaler.n_samples_seen_),
                          'mean_log1p': float(scaler.mean_[0]), 'scale_log1p': float(scaler.scale_[0])},
        'target_transform': 'Training-standardized log1p KZT; ANNRegressor converts predictions back to finite nonnegative KZT.',
        'aggregation': 'Equal arithmetic mean of every fixed seed; no seed subsets are searched.',
        'baseline_validation_metrics': baseline, 'candidate_validation_metrics': candidate_metrics,
        'members': members,
        'member_best_epochs': {key: [member['checkpoints'][key]['best_epoch'] for member in members] for key in pools},
        'member_epochs_run': [member['epochs_run'] for member in members],
        'loss': 'Ordinary MSE of training-standardized log1p targets, without optimizer one-half factor or L2 term.',
        'checkpoint_selection': 'Strictly lower validation original-KZT MAE or 0.5*(MAE/baselineMAE + missFraction/baselineMissFraction); ties keep the first checkpoint. Stop 45 epochs after both best epochs stop improving under the default protocol.',
        'checkpoint_selection_caveat': 'Validation is reused to select every checkpoint and the blend. Its reported gains and convergence are selection-dependent, not independent generalization evidence. The historical test was already known and any final evaluation is retrospective.',
        'source': {'csv': 'kolesa_cars_2025.csv', 'source_url': data.SOURCE,
                   'features': ('Original per-row vehicle fields and literal pre-КПП steering clauses'
                                + (' plus explicitly declared fixed-snapshot age/usage proxies' if additional_numeric else '')
                                + ' only; price, price_raw, ad_id and link are not predictors.')},
        'artifacts': {'history_csv': 'robust_ann_history.csv', 'history_plot': 'robust_ann_history.png',
                      'models': model_files, 'report_json': 'robust_ann_report.json'},
        'test_data_used': False, 'test_data_policy': 'Test data is not accepted or evaluated by this API.',
    }
    (destination / 'robust_ann_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return models, report


def select_dual_blend(validation_predictions, y_val, baseline_name, ann_names,
                      forest_name, weights=(.25, .5, .75), *, include_baseline_blends=False):
    """Select from the original nine candidates or an explicit expanded grid.

    Both validation MAE and within-20% hit rate must improve strictly over the
    baseline. Among eligible candidates minimize the equally weighted
    normalized MAE/miss score; if none qualify, retain the baseline. With
    ``include_baseline_blends=True``, each pool is also blended with the same
    baseline at every declared weight (15 candidates with the default grid).
    Normalization always uses that baseline, including for baseline mixtures.
    """
    target = _target(y_val, 'Validation')
    ann_names = tuple(ann_names)
    if len(ann_names) != 2 or len(set((baseline_name, *ann_names, forest_name))) != 4:
        raise ValueError('Specify distinct baseline, two ANN pool names and forest name')
    declared_weights = tuple(float(weight) for weight in weights)
    if not declared_weights or any(not np.isfinite(weight) or not 0 < weight < 1 for weight in declared_weights):
        raise ValueError('Blend weights must be finite and strictly between zero and one')
    predictions = {name: _prediction(validation_predictions[name], target, name)
                   for name in (baseline_name, *ann_names, forest_name)}
    baseline = _reference(target, predictions[baseline_name])
    candidates = []

    def record(name, components, component_weights, prediction):
        values = _metrics(target, prediction, baseline)
        candidates.append({
            'name': name, 'candidate': name, 'components': list(components), 'weights': list(component_weights),
            **values, 'validation_mae_kzt': values['mae_kzt'],
            'validation_within_20pct': values['within_20pct'],
            'eligible': bool(values['mae_kzt'] < baseline['mae_kzt']
                             and values['within_20pct'] > baseline['within_20pct']),
        })

    record(baseline_name, [baseline_name], [1.], predictions[baseline_name])
    for name in ann_names:
        record(name, [name], [1.], predictions[name])
    for name in ann_names:
        for other in ((forest_name, baseline_name) if include_baseline_blends else (forest_name,)):
            for weight in declared_weights:
                record(f'{name} {weight:g} + {other} {1 - weight:g}',
                       [name, other], [weight, 1 - weight],
                       weight * predictions[name] + (1 - weight) * predictions[other])
    eligible = [row for row in candidates if row['eligible']]
    selected = min(eligible, key=lambda row: row['dual_score']) if eligible else candidates[0]
    return {
        'selected_name': selected['name'], 'selected_components': selected['components'],
        'weights': selected['weights'], 'validation_mae_kzt': selected['mae_kzt'],
        'validation_within_20pct': selected['within_20pct'],
        'mae_kzt': selected['mae_kzt'], 'within_20pct': selected['within_20pct'],
        'median_ape': selected['median_ape'], 'dual_score': selected['dual_score'],
        'candidates': candidates, 'eligible_count': len(eligible),
        'baseline_validation_metrics': baseline,
        'selection': 'Strict validation MAE decrease and within-20% increase versus baseline; then lowest 0.5*(MAE/baselineMAE + missFraction/baselineMissFraction). Baseline retained if none; exact score ties retain the first declared candidate.',
        'declared_blend_weights': list(declared_weights),
        'include_baseline_blends': bool(include_baseline_blends),
        'test_data_used': False, 'test_data_policy': 'No test targets or predictions accepted.',
        'checkpoint_selection_caveat': 'Validation has already selected pool checkpoints and now selects the mixture; results are selection-dependent. Historical-test reuse prevents an independent generalization claim.',
    }
