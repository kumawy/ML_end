"""Bounded, predeclared TRAIN-only ANN feature ablation.

Run from any checkout with ``python run_utility_check.py --project-source ../code``.
The source project is read-only. This experiment does not select or save a final
model, and never extracts validation/test targets or predicts those rows.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


REPRESENTATIONS = (
    ('base', (), ()),
    ('base_plus_age_usage', ('log_vehicle_age', 'log_annual_mileage_proxy'), ()),
    ('base_plus_lexical_family', (), ('model_family_lexical',)),
    ('base_plus_literal_claims', ('alloy_wheels_claim', 'sunroof_claim'),
     ('color_literal', 'upholstery_literal')),
    ('base_plus_all_seven', ('log_vehicle_age', 'log_annual_mileage_proxy',
     'alloy_wheels_claim', 'sunroof_claim'),
     ('model_family_lexical', 'color_literal', 'upholstery_literal')),
)
EXPECTED_NUMERIC = ('log_vehicle_age', 'log_annual_mileage_proxy',
                    'alloy_wheels_claim', 'sunroof_claim')
EXPECTED_CATEGORICAL = ('model_family_lexical', 'color_literal', 'upholstery_literal')
NETWORK_CONFIG = {
    'hidden_layer_sizes': [128, 64], 'activation': 'relu', 'solver': 'adam',
    'alpha': .001, 'batch_size': 64, 'learning_rate_init': .001,
    'random_state': 42, 'shuffle': True, 'early_stopping': False,
    'max_iter': 1, 'tol': 0.0,
}
MAX_EPOCHS = 150
PATIENCE = 25


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError('Predictions must be finite and align with targets.')
    if np.any(actual <= 0) or np.any(predicted <= 0):
        raise ValueError('This utility check requires finite positive KZT values.')
    relative = np.abs(actual - predicted) / actual
    return {'mae_kzt': float(np.mean(np.abs(actual - predicted))),
            'within_20pct': float(np.mean(relative <= .20)),
            'median_ape': float(np.median(relative))}


def preprocessor(base_numeric, base_categorical, extra_numeric, extra_categorical):
    branches = [
        ('base_num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), list(base_numeric)),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                     min_frequency=2, sparse_output=False)),
        ]), [*base_categorical, *extra_categorical]),
    ]
    if extra_numeric:
        # Positive claim = 1, unmentioned = NaN. Indicators preserve the
        # unmentioned state after fold-local median imputation; empty columns
        # stay present rather than silently disappearing in rare-claim folds.
        branches.append(('extra_num', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True,
                                     keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), list(extra_numeric)))
    return ColumnTransformer(branches, sparse_threshold=0, remainder='drop')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-source', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'code')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    source = args.project_source.resolve()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'report.json').exists():
        raise FileExistsError('Completed output exists; choose a fresh output directory.')
    sys.path.insert(0, str(source))
    data = importlib.import_module('data')
    robust_ann = importlib.import_module('robust_ann')
    structured = importlib.import_module('structured_features')
    extra = importlib.import_module('dataset_features')
    ANNRegressor = importlib.import_module('ann').ANNRegressor
    if tuple(extra.EXTRA_NUMERIC) != EXPECTED_NUMERIC:
        raise ValueError('New numeric feature contract differs from predeclared protocol.')
    if tuple(extra.EXTRA_CATEGORICAL) != EXPECTED_CATEGORICAL:
        raise ValueError('New categorical feature contract differs from predeclared protocol.')
    if len(robust_ann.ROBUST_FEATURES) != 13:
        raise ValueError('Expected 13 corrected-plus-steering base fields.')

    # Existing deterministic cleaning and split stay exactly as in the source
    # project. Only the original TRAIN labels are extracted below.
    frame, audit = data.load_data(source)
    split = data.split_data(frame)
    if len(split['train']) != 1787:
        raise ValueError('Expected the original 1,787 TRAIN rows.')
    frozen = pd.read_csv(source / 'results' / 'split_manifest.csv',
                         usecols=['source_row', 'split'])
    for name, rows in split.items():
        if list(map(int, rows)) != frozen.loc[frozen['split'].eq(name), 'source_row'].tolist():
            raise ValueError(f'Original split ordering mismatch: {name}')
    training = frame.loc[split['train']].copy()
    del frame
    target = training['price'].to_numpy(dtype=float)
    if not np.isfinite(target).all() or np.any(target <= 0):
        raise ValueError('Original training targets must be finite and positive.')
    identifiers = pd.DataFrame({'train_position': np.arange(len(training)),
                               'source_row': training.index.to_numpy(dtype=int),
                               'ad_id': training['ad_id'].astype(str).to_numpy()})
    # Metadata and target columns are never passed to either feature helper or
    # fitted transformer. Every representation has an explicit field allowlist.
    feature_source = training.drop(columns=['price', 'price_raw', 'link', 'ad_id'])
    del training
    features = extra.add_dataset_features(structured.add_structured_features(feature_source))
    required = [*robust_ann.ROBUST_FEATURES, *EXPECTED_NUMERIC, *EXPECTED_CATEGORICAL]
    if set(required) - set(features.columns):
        raise ValueError('Missing predeclared features.')
    features = features.loc[:, required].copy()
    if set(features.columns) != set(required) or features.index.has_duplicates:
        raise ValueError('Predictor allowlist or original row integrity failed.')
    del feature_source

    folds, roles = [], []
    for fold_id, (outer_train, outer_holdout) in enumerate(
            KFold(n_splits=3, shuffle=True, random_state=42).split(features), 1):
        inner_fit, inner_checkpoint = next(ShuffleSplit(
            n_splits=1, test_size=.20, random_state=20261107).split(outer_train))
        inner_fit, inner_checkpoint = outer_train[inner_fit], outer_train[inner_checkpoint]
        assert not (set(inner_fit) & set(inner_checkpoint))
        assert not (set(outer_train) & set(outer_holdout))
        folds.append((fold_id, inner_fit, inner_checkpoint, outer_holdout))
        for role, positions in [('inner_train', inner_fit),
                                ('inner_checkpoint', inner_checkpoint),
                                ('outer_holdout', outer_holdout)]:
            members = identifiers.iloc[positions].copy()
            members.insert(0, 'fold', fold_id)
            members['role'] = role
            roles.append(members)

    relevant_files = ['data.py', 'ann.py', 'robust_ann.py', 'structured_features.py',
                      'dataset_features.py', 'kolesa_cars_2025.csv',
                      'results/split_manifest.csv']
    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'project_source_relative_to_output': os.path.relpath(source, destination),
        'python_version': platform.python_version(), 'numpy_version': np.__version__,
        'pandas_version': pd.__version__, 'sklearn_version': sklearn.__version__,
        'source_sha256': {name: sha256(source / name) for name in relevant_files},
        'script_sha256': sha256(__file__),
        'original_source_git_blob': audit['source_git_blob_sha'],
        'original_split_counts': {key: len(value) for key, value in split.items()},
        'extracted_target_split': 'train only',
        'validation_test_targets_extracted': False,
        'validation_test_predictions_made': False,
        'new_price_based_filtering': False,
    }
    protocol = {
        'purpose': 'Descriptive controlled feature ablation within original TRAIN only.',
        'representations_in_fixed_order': [
            {'name': name, 'base_features': list(robust_ann.ROBUST_FEATURES),
             'extra_numeric': list(num), 'extra_categorical': list(cat)}
            for name, num, cat in REPRESENTATIONS],
        'outer': {'method': 'KFold', 'n_splits': 3, 'shuffle': True, 'random_state': 42},
        'inner': {'method': 'ShuffleSplit', 'n_splits': 1, 'test_size': .20,
                  'random_state': 20261107, 'selection': 'lowest inner checkpoint MAE in KZT'},
        'network': NETWORK_CONFIG, 'maximum_epochs': MAX_EPOCHS, 'patience': PATIENCE,
        'maximum_fits': 15, 'threadpool_limit': 1,
        'preprocessing': {'fit_rows': 'inner_train only', 'base_numeric': 'median, StandardScaler',
                          'extra_numeric': 'median, add_indicator=True, keep_empty_features=True, StandardScaler',
                          'categorical': 'unknown constant, one-hot min_frequency=2, handle_unknown=infrequent_if_exist'},
        'target_transform': 'inner_train-fitted standardized log1p; existing ANNRegressor inverse in KZT',
        'claim_semantics': 'Positive literal claim is 1; unmentioned is NaN, retained through indicators.',
        'excluded_predictors': 'All quality flags, metadata, price/price_raw, listing IDs, and URLs.',
        'outer_evaluation': 'One prediction per held-out TRAIN row per representation, after inner checkpoint selection.',
        'reported_metrics': ['mae_kzt', 'within_20pct', 'median_ape'],
        'interpretation': 'One seed and three outer folds; not directly comparable with main validation/test results or the five-seed final ANN ensemble.',
        'selection_policy': 'No OOF-driven tuning, hypotheses, ranking, winner selection, model saving, or final-model promotion; all five outcomes including negatives retained.',
        'provenance': provenance,
    }
    # This frozen protocol and every split assignment are persisted before the
    # first fit. No later result changes any representation or checkpoint rule.
    write_json(destination / 'protocol_before_evaluation.json', protocol)
    pd.concat(roles, ignore_index=True).to_csv(destination / 'fold_roles.csv', index=False)
    write_json(destination / 'provenance.json', provenance)

    histories, fold_results, oof_rows = [], [], []
    with threadpool_limits(limits=1):
        for name, extra_numeric, extra_categorical in REPRESENTATIONS:
            for fold_id, fit_positions, checkpoint_positions, holdout_positions in folds:
                prep = preprocessor(robust_ann.ROBUST_NUMERIC, robust_ann.ROBUST_CATEGORICAL,
                                    extra_numeric, extra_categorical)
                fit_x = prep.fit_transform(features.iloc[fit_positions])
                checkpoint_x = prep.transform(features.iloc[checkpoint_positions])
                if not np.isfinite(fit_x).all() or not np.isfinite(checkpoint_x).all():
                    raise ValueError('Nonfinite preprocessed inner predictors.')
                fit_y, checkpoint_y = target[fit_positions], target[checkpoint_positions]
                scaler = StandardScaler().fit(np.log1p(fit_y).reshape(-1, 1))
                fit_z = scaler.transform(np.log1p(fit_y).reshape(-1, 1)).ravel()
                checkpoint_z = scaler.transform(np.log1p(checkpoint_y).reshape(-1, 1)).ravel()
                network = MLPRegressor(**{**NETWORK_CONFIG, 'hidden_layer_sizes': (128, 64)})
                best_mae, best_epoch, best_network = float('inf'), 0, None
                fold_history = []
                for epoch in range(1, MAX_EPOCHS + 1):
                    network.partial_fit(fit_x, fit_z)
                    wrapped = ANNRegressor(network, scaler)
                    checkpoint_metrics = metrics(checkpoint_y, wrapped.predict(checkpoint_x))
                    row = {
                        'representation': name, 'fold': fold_id, 'epoch': epoch,
                        'train_scaled_log_mse': float(np.mean((fit_z - network.predict(fit_x)) ** 2)),
                        'checkpoint_scaled_log_mse': float(np.mean((checkpoint_z - network.predict(checkpoint_x)) ** 2)),
                        'train_mae_kzt': metrics(fit_y, wrapped.predict(fit_x))['mae_kzt'],
                        **{f'checkpoint_{key}': value for key, value in checkpoint_metrics.items()},
                    }
                    if not all(np.isfinite(value) for key, value in row.items()
                               if key not in ('representation',)):
                        raise FloatingPointError('Nonfinite inner training metrics.')
                    fold_history.append(row)
                    if checkpoint_metrics['mae_kzt'] < best_mae:
                        best_mae, best_epoch, best_network = checkpoint_metrics['mae_kzt'], epoch, deepcopy(network)
                    if epoch - best_epoch >= PATIENCE:
                        break
                for row in fold_history:
                    row['is_selected_epoch'] = row['epoch'] == best_epoch
                histories.extend(fold_history)
                # Outer inputs are transformed and predicted only once, after
                # checkpoint selection. Outer labels play no role in training.
                holdout_x = prep.transform(features.iloc[holdout_positions])
                predictions = ANNRegressor(best_network, scaler).predict(holdout_x)
                holdout_metrics = metrics(target[holdout_positions], predictions)
                fold_results.append({
                    'representation': name, 'fold': fold_id, 'selected_epoch': best_epoch,
                    'epochs_run': epoch, 'stop_reason': 'patience' if epoch - best_epoch >= PATIENCE else 'maximum_epochs',
                    'inner_train_rows': len(fit_positions), 'inner_checkpoint_rows': len(checkpoint_positions),
                    'outer_holdout_rows': len(holdout_positions), 'transformed_features': fit_x.shape[1],
                    'target_scaler_n_samples_seen': int(scaler.n_samples_seen_),
                    'target_scaler_mean_log1p': float(scaler.mean_[0]),
                    'target_scaler_scale_log1p': float(scaler.scale_[0]),
                    'selected_inner_mae_kzt': best_mae, **holdout_metrics,
                })
                predicted_rows = identifiers.iloc[holdout_positions].copy()
                predicted_rows.insert(0, 'representation', name)
                predicted_rows.insert(1, 'fold', fold_id)
                predicted_rows['actual_price_kzt'] = target[holdout_positions]
                predicted_rows['predicted_price_kzt'] = predictions
                oof_rows.append(predicted_rows)
                # Incremental audit artifacts survive interruption; a final
                # report is written only after all 15 fits finish.
                pd.DataFrame(histories).to_csv(destination / 'fold_histories.csv', index=False)
                pd.DataFrame(fold_results).to_csv(destination / 'fold_metrics.csv', index=False)
                pd.concat(oof_rows, ignore_index=True).to_csv(destination / 'oof_predictions.csv', index=False)
                print(f'{name} fold={fold_id}: epoch={best_epoch}/{epoch}, MAE={holdout_metrics["mae_kzt"]:.0f}', flush=True)

    oof = pd.concat(oof_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_results)
    aggregate = []
    for name, _, _ in REPRESENTATIONS:
        rows = oof.loc[oof.representation.eq(name)]
        if len(rows) != len(target) or set(rows.train_position) != set(range(len(target))):
            raise ValueError('OOF coverage must be exactly one prediction per original TRAIN row.')
        result = metrics(rows.actual_price_kzt.to_numpy(), rows.predicted_price_kzt.to_numpy())
        result.update({'representation': name, 'rows': len(rows), 'folds': 3})
        for metric in ('mae_kzt', 'within_20pct', 'median_ape'):
            result[f'fold_mean_{metric}'] = float(fold_metrics.loc[fold_metrics.representation.eq(name), metric].mean())
            result[f'fold_std_{metric}'] = float(fold_metrics.loc[fold_metrics.representation.eq(name), metric].std(ddof=1))
        aggregate.append(result)
    baseline = aggregate[0]
    baseline_folds = fold_metrics.loc[fold_metrics.representation.eq('base')].set_index('fold')
    for result in aggregate:
        result['delta_mae_vs_base_kzt'] = result['mae_kzt'] - baseline['mae_kzt']
        result['delta_within_20pct_vs_base_pp'] = 100 * (result['within_20pct'] - baseline['within_20pct'])
        result['delta_median_ape_vs_base_pp'] = 100 * (result['median_ape'] - baseline['median_ape'])
        own = fold_metrics.loc[fold_metrics.representation.eq(result['representation'])].set_index('fold')
        result['folds_with_lower_mae_than_base'] = int((own.mae_kzt < baseline_folds.mae_kzt).sum())
        result['folds_with_higher_within_20pct_than_base'] = int((own.within_20pct > baseline_folds.within_20pct).sum())
    pd.DataFrame(aggregate).to_csv(destination / 'aggregate_metrics.csv', index=False)
    fold_metrics.loc[:, ['representation', 'fold', 'selected_epoch', 'epochs_run', 'stop_reason']].to_csv(
        destination / 'selected_epochs.csv', index=False)
    write_json(destination / 'report.json', {
        'status': 'complete', 'fits_completed': len(fold_results), 'training_rows': len(target),
        'aggregate_metrics_in_predeclared_order': aggregate,
        'interpretation': protocol['interpretation'],
        'selection_policy': protocol['selection_policy'],
        'protocol_sha256': sha256(destination / 'protocol_before_evaluation.json'),
        'provenance': provenance,
        'artifacts': ['protocol_before_evaluation.json', 'provenance.json', 'fold_roles.csv',
                      'fold_histories.csv', 'fold_metrics.csv', 'selected_epochs.csv',
                      'oof_predictions.csv', 'aggregate_metrics.csv'],
    })


if __name__ == '__main__':
    main()
