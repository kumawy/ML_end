#!/usr/bin/env python3
"""Independently verify completed TRAIN-only simplicity evidence without fitting.

This program never imports simplicity_review or calls any estimator. The common
source loader reproduces eligibility and identities; only TRAIN targets are
extracted. Saved predictions, not old validation/test predictions, determine all
reported metrics and the recommendation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


NAMES = ('single_mae', 'single_log', 'single_forest', 'historical_recipe')
COUNTS = {'single_mae': 1, 'single_log': 1, 'single_forest': 2, 'historical_recipe': 11}
BOUNDARIES = np.array([3_600_000., 6_700_000., 11_500_000.])
PROTOCOL_SHA256 = '5e58f6a8518327e05e221145447e3a03998d257e42cab7205dc2f6adf674508d'
SOURCE_SHA256 = 'b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468'
ORIGINAL_HASHES = {
    'results/selected_model.joblib': '00a8820cfd5b9deee4f7993a1bbcc595ae01a4f9db9c9fb371715478ef557296',
    'results/model_comparison.csv': '782511a39ddd2cc7b8127d6e390432f6fc06327ec7651ce044cebea9b8a80080',
    'results/all_test_predictions.csv': '095696d34c5f50db121eaa08a2ddb10fdc01d9cbf3e85e2897affdaf3cb3bfc1',
    'results/selected_model_test_predictions.csv': '48b6394efccceb70d1563d34ab6eeb24d85ed6688212708de9c2e6738701fd45',
}
CONFIGURATION = {'single_epochs': 300, 'single_patience': 45,
                 'reference_epochs': 400, 'reference_patience': 40,
                 'pool_epochs': 300, 'pool_patience': 45}


class VerificationError(ValueError):
    """A saved assertion did not match independently reconstructed evidence."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key {key!r} in {path}')
            result[key] = value
        return result

    def invalid(value):
        raise VerificationError(f'Nonfinite JSON value {value} in {path}')

    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=unique, parse_constant=invalid)


def local_file(directory, name):
    require(isinstance(name, str), 'Artifact path must be a string')
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts,
            f'Artifact path is not safely relative: {name}')
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory), f'Artifact escapes directory: {name}')
    require(path.is_file(), f'Missing artifact: {path}')
    return path


def match(saved, expected, path):
    """Compare every independently expected field; tolerate only float I/O noise."""
    if isinstance(expected, dict):
        require(isinstance(saved, dict), f'{path} must be an object')
        for key, value in expected.items():
            require(key in saved, f'Missing {path}.{key}')
            match(saved[key], value, f'{path}.{key}')
    elif isinstance(expected, list):
        require(isinstance(saved, list) and len(saved) == len(expected), f'{path} length mismatch')
        for number, (actual, wanted) in enumerate(zip(saved, expected)):
            match(actual, wanted, f'{path}[{number}]')
    elif isinstance(expected, bool):
        require(isinstance(saved, bool) and saved is expected, f'{path}: {saved!r} != {expected!r}')
    elif isinstance(expected, int):
        require(not isinstance(saved, bool) and isinstance(saved, (int, float))
                and saved == expected, f'{path}: {saved!r} != {expected!r}')
    elif isinstance(expected, float):
        require(not isinstance(saved, bool) and isinstance(saved, (int, float))
                and np.isfinite(saved) and np.isclose(saved, expected, rtol=1e-11, atol=1e-8),
                f'{path}: {saved!r} != {expected!r}')
    else:
        require(saved == expected, f'{path}: {saved!r} != {expected!r}')


def integer_values(values, name):
    numbers = pd.to_numeric(values, errors='raise').to_numpy(dtype=float)
    require(np.isfinite(numbers).all() and (numbers == np.floor(numbers)).all(),
            f'{name} must contain finite integers')
    return numbers.astype(np.int64)


def recompute_metrics(target, prediction):
    """Compute counts directly from pooled errors with fixed inclusive cutoffs."""
    error = np.abs(target - prediction)
    ape = error / target

    def metrics(indices):
        n = int(np.count_nonzero(indices))
        hits = int(np.count_nonzero(ape[indices] <= .20))
        return {'n': n, 'mae_kzt': float(error[indices].mean()) if n else None,
                'within_20pct_count': hits, 'within_20pct': hits / n if n else None,
                'median_ape': float(np.median(ape[indices])) if n else None}

    bands = [(target <= BOUNDARIES[0]),
             ((target > BOUNDARIES[0]) & (target <= BOUNDARIES[1])),
             ((target > BOUNDARIES[1]) & (target <= BOUNDARIES[2])),
             (target > BOUNDARIES[2])]
    return {**metrics(np.ones(len(target), dtype=bool)),
            'affordable': metrics(target <= BOUNDARIES[1]),
            'bands': {f'Q{i + 1}': metrics(mask) for i, mask in enumerate(bands)}}


def independent_decision(summaries):
    baseline = summaries['historical_recipe']
    candidates, eligible = [], []
    for name in NAMES[:-1]:
        row = summaries[name]
        q1 = row['bands']['Q1']['within_20pct_count']
        q2 = row['bands']['Q2']['within_20pct_count']
        d1 = baseline['bands']['Q1']['within_20pct_count']
        d2 = baseline['bands']['Q2']['within_20pct_count']
        gates = {
            'mae_within_3pct': row['mae_kzt'] <= baseline['mae_kzt'] * 1.03,
            'overall_hits_not_worse': row['within_20pct_count'] >= baseline['within_20pct_count'],
            'q1_hits_not_worse': q1 >= d1, 'q2_hits_not_worse': q2 >= d2,
            'affordable_band_strict_hit_gain': q1 > d1 or q2 > d2,
            'affordable_median_ape_not_worse': row['affordable']['median_ape'] <= baseline['affordable']['median_ape'],
        }
        passes = all(gates.values())
        candidates.append({'name': name, 'eligible': passes, 'gates': gates,
                           'estimator_count': COUNTS[name]})
        if passes:
            eligible.append(name)
    ranked = sorted(eligible, key=lambda name: (
        COUNTS[name], summaries[name]['affordable']['median_ape'], summaries[name]['mae_kzt'], name))
    return {'selected_name': ranked[0] if ranked else 'historical_recipe',
            'eligible_names': eligible, 'candidates': candidates,
            'ranking': 'estimator count, affordable median APE, global MAE, name',
            'decision': ('Refit a separate development candidate' if ranked else
                         'No tested simplification met the joint preference; retain historical recipe'),
            'status': 'development/not_independent'}


def training_identity(source):
    # data.py contains deterministic row eligibility and the original split.
    # No candidate module is imported and no old holdout target vector is formed.
    spec = importlib.util.spec_from_file_location('_simplicity_verifier_source_data', source / 'data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame, audit = module.load_data(source)
    split = module.split_data(frame)
    manifest = pd.read_csv(source / 'results/split_manifest.csv', dtype={'ad_id': str})
    require(set(manifest.columns) == {'source_row', 'ad_id', 'split'}, 'Original split manifest schema differs')
    require(len(manifest) == 2553 and not manifest.source_row.duplicated().any()
            and not manifest.ad_id.duplicated().any(), 'Original split identities are not unique')
    require(set(manifest['split']) == {'train', 'validation', 'test'}, 'Unexpected original split')
    match({name: len(rows) for name, rows in split.items()},
          {'train': 1787, 'validation': 383, 'test': 383}, 'original_split_counts')
    for name, rows in split.items():
        saved = manifest.loc[manifest['split'].eq(name)]
        require(saved.source_row.tolist() == list(map(int, rows)), f'Original {name} row ordering mismatch')
        require(saved.ad_id.tolist() == frame.loc[rows, 'ad_id'].astype(str).tolist(),
                f'Original {name} ad IDs mismatch source')
    training = frame.loc[split['train'], ['ad_id', 'price']].sort_index().copy()
    del frame
    require(audit['source_sha256'] == SOURCE_SHA256, 'Common loader source hash mismatch')
    target = training.price.to_numpy(dtype=float)
    require(np.isfinite(target).all() and (target > 0).all(), 'TRAIN labels must be finite positive KZT')
    return training, target, manifest


def verify_checkpoint_history(directory, report_name, prefix, criteria, cap, patience, inner_target, checkpoint_rows):
    report = read_json(directory / report_name)
    history = pd.read_csv(directory / f'{prefix}_history.csv')
    epochs = integer_values(history.epoch, f'{prefix} epochs')
    require(1 <= len(history) <= cap and epochs.tolist() == list(range(1, len(history) + 1)),
            f'{prefix}: noncontiguous or out-of-range epochs')
    values = history[list(criteria.values())].to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values >= 0).all(), f'{prefix}: invalid checkpoint losses')
    best = {name: int(np.argmin(history[column].to_numpy())) + 1 for name, column in criteria.items()}
    match(report, {'status': 'development/not_independent', 'training_rows': len(inner_target),
                   'checkpoint_rows': checkpoint_rows, 'epochs_run': len(history), 'max_epochs': cap,
                   'patience': patience, 'min_frequency': 2,
                   'selected_epochs': best,
                   'selected_values': {name: float(history[column].iloc[best[name] - 1])
                                       for name, column in criteria.items()},
                   'preprocessing_fit_split': 'inner training only'}, str(directory / report_name))
    verify_scaler(report['target_scaler'], inner_target, prefix)
    require(len(history) == cap or len(history) - max(best.values()) == patience,
            f'{prefix}: stopped outside frozen cap/patience rule')
    for end in range(1, len(history)):
        earlier_best = max(int(np.argmin(history[column].iloc[:end].to_numpy())) + 1
                           for column in criteria.values())
        require(end - earlier_best < patience, f'{prefix}: continued beyond first patience stop')
    return best


def verify_scaler(saved, target, label):
    log = np.log1p(target)
    scale = float(np.std(log, ddof=0)) or 1.
    match(saved, {'n_samples_seen': len(target), 'mean_log1p': float(log.mean()),
                  'scale_log1p': scale}, f'{label}.target_scaler')


def verify_pool(directory, inner_target, checkpoint_rows):
    report = read_json(directory / 'robust_ann_report.json')
    history = pd.read_csv(directory / 'robust_ann_history.csv')
    match(report, {'training_rows': len(inner_target), 'validation_rows': checkpoint_rows,
                   'preprocessing_fit_split': 'training only', 'test_data_used': False,
                   'configuration': {'seeds': [11, 23, 42, 71, 101], 'max_epochs': 300,
                                     'patience': 45, 'min_frequency': 2}}, str(directory))
    verify_scaler(report['target_scaler'], inner_target, str(directory))
    members = report['members']
    require([member['seed'] for member in members] == [11, 23, 42, 71, 101], f'{directory}: fixed seeds differ')
    require(set(integer_values(history.seed, 'pool seed')) == {11, 23, 42, 71, 101}, f'{directory}: history seeds differ')
    selected = {'mae_kzt': [], 'dual_score': []}
    for member in members:
        own = history.loc[history.seed.eq(member['seed'])]
        require(integer_values(own.epoch, 'pool epochs').tolist() == list(range(1, len(own) + 1))
                and 1 <= len(own) <= 300, f'{directory}: invalid seed trajectory')
        values = own[['mae_kzt', 'dual_score']].to_numpy(dtype=float)
        require(np.isfinite(values).all() and (values >= 0).all(), f'{directory}: invalid pool losses')
        best = {key: int(np.argmin(own[key].to_numpy())) + 1 for key in selected}
        for key, epoch in best.items():
            selected[key].append(epoch)
            match(member['checkpoints'][key], {'best_epoch': epoch, 'best_value': float(own[key].iloc[epoch - 1])},
                  f'{directory}.seed{member["seed"]}.{key}')
        require(len(own) == 300 or len(own) - max(best.values()) == 45,
                f'{directory}: invalid both-checkpoint stopping')
        match(member, {'epochs_run': len(own), 'stopped_after_both_checkpoints_stale': len(own) - max(best.values()) >= 45},
              f'{directory}.seed{member["seed"]}')
        for end in range(1, len(own)):
            earlier_best = max(int(np.argmin(own[key].iloc[:end].to_numpy())) + 1 for key in selected)
            require(end - earlier_best < 45, f'{directory}: trajectory continued beyond patience')
    match(report['member_best_epochs'], selected, f'{directory}.member_best_epochs')
    return selected


def verify(source, output):
    provenance = read_json(output / 'provenance.json')
    summary = read_json(output / 'summary.json')
    before = read_json(output / 'preserved_artifacts_before.json')
    after = read_json(output / 'preserved_artifacts_after.json')
    require(sha256(source / 'kolesa_cars_2025.csv') == SOURCE_SHA256, 'Source CSV differs from frozen original')
    require(sha256(source / 'experiments/simplicity_review/PROTOCOL.md') == PROTOCOL_SHA256,
            'Current protocol differs from independently frozen protocol')
    require(sha256(output / 'protocol_before_evaluation.md') == PROTOCOL_SHA256,
            'Pre-fit protocol copy differs from independently frozen protocol')
    match(provenance, {'status': 'prepared_before_any_fitting', 'evaluation_status': 'development/not_independent',
                       'source_csv': 'kolesa_cars_2025.csv', 'source_sha256': SOURCE_SHA256,
                       'source_git_blob_sha': '01e57295927f91ffd7996745ce6a867b6d41b0f7',
                       'protocol_sha256': PROTOCOL_SHA256, 'training_rows': 1787,
                       'excluded_validation_rows': 383, 'excluded_test_rows': 383,
                       'band_upper_kzt': BOUNDARIES.tolist(), 'configuration': CONFIGURATION,
                       'component_counts': COUNTS,
                       'outer_splitter': {'n_splits': 5, 'shuffle': True, 'random_state': 20261107},
                       'inner_splitter': {'name': 'train_test_split', 'test_size': .2,
                                          'stratify': 'fixed price bands', 'random_state': 20261108}}, 'provenance')
    code_hashes = provenance['code_sha256']
    required_code = {path.name for path in source.glob('*.py')}
    required_code.add('experiments/simplicity_review/run_simplicity_review.py')
    require(set(code_hashes) == required_code, 'Frozen code hash inventory omits or adds source modules')
    for name, expected in code_hashes.items():
        require(sha256(local_file(source, name)) == expected, f'Frozen code changed: {name}')
    require(isinstance(before, dict) and bool(before) and before == after
            and before == provenance['protected_artifact_hashes'], 'Protected before/after inventories differ')
    for name, expected in ORIGINAL_HASHES.items():
        require(before.get(name) == expected, f'Before-run historical baseline differs: {name}')
    for name, expected in before.items():
        require(sha256(local_file(source, name)) == expected, f'Historical artifact changed: {name}')
    # New experiment directories may have appeared since the inventory froze,
    # and a completed run may be copied elsewhere for auditing. Require the
    # established top-level benchmark files and reference/course artifacts;
    # every nested path in the frozen inventory is hash-checked above as well.
    required_preserved = [*source.glob('*.joblib'), *(source / 'results').glob('*'),
                          *(source / 'reference').rglob('*')]
    for path in required_preserved:
        if path.is_file() and path.suffix in {'.csv', '.json', '.joblib', '.pkl'}:
            require(path.relative_to(source).as_posix() in before,
                    f'Historical metric/model omitted from protection: {path}')

    training, target, original_manifest = training_identity(source)
    rows = training.index.to_numpy(dtype=int)
    ids = training.ad_id.astype(str).tolist()
    match(provenance['train_source_rows'], rows.tolist(), 'provenance.train_source_rows')
    match(provenance['train_ad_ids'], ids, 'provenance.train_ad_ids')
    predictions = pd.read_csv(output / 'oof_predictions.csv', dtype={'ad_id': str})
    identity_columns = ['source_row', 'ad_id', 'fold', 'actual_price_kzt']
    require(predictions.columns.tolist() == identity_columns + list(NAMES), 'OOF CSV column schema/order mismatch')
    require(len(predictions) == 1787 and not predictions.source_row.duplicated().any()
            and not predictions.ad_id.duplicated().any() and not predictions.ad_id.isna().any(),
            'OOF identities must appear exactly once')
    require(integer_values(predictions.source_row, 'OOF source_row').tolist() == rows.tolist(), 'OOF rows differ from sorted original TRAIN')
    require(predictions.ad_id.tolist() == ids, 'OOF ad IDs misaligned with source rows')
    old_holdout = original_manifest.loc[~original_manifest['split'].eq('train')]
    require(not (set(predictions.source_row) & set(old_holdout.source_row))
            and not (set(predictions.ad_id) & set(old_holdout.ad_id)), 'OOF contains original VAL/TEST identities')
    numeric = predictions[['actual_price_kzt', *NAMES]].to_numpy(dtype=float)
    require(np.isfinite(numeric).all() and (numeric[:, 0] > 0).all() and (numeric[:, 1:] >= 0).all(),
            'OOF labels/predictions contain nonfinite or negative values')
    require(np.array_equal(numeric[:, 0], target), 'OOF labels differ from original source TRAIN labels')
    folds = pd.read_csv(output / 'folds.csv', dtype={'ad_id': str})
    pd.testing.assert_frame_equal(folds, predictions[identity_columns], check_exact=True)
    assigned = integer_values(predictions.fold, 'OOF fold')
    require(set(assigned) == {1, 2, 3, 4, 5}, 'Outer folds must be one-based 1..5')

    plans, fold_results, selected_epochs = [], [], []
    price_bands = np.digitize(target, BOUNDARIES, right=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261107)
    require(len(summary['folds']) == 5, 'Expected five fold summaries')
    for number, (develop, outer) in enumerate(splitter.split(rows, price_bands), 1):
        inner, checkpoint = train_test_split(develop, test_size=.2, stratify=price_bands[develop], random_state=20261108)
        partitions = [set(rows[part]) for part in (inner, checkpoint, outer)]
        require(not partitions[0] & partitions[1] and not partitions[0] & partitions[2]
                and not partitions[1] & partitions[2] and set.union(*partitions) == set(rows),
                f'Fold {number} partitions leak or omit TRAIN identities')
        plan = {'fold': number, 'inner_train_source_rows': rows[inner].tolist(),
                'inner_checkpoint_source_rows': rows[checkpoint].tolist(),
                'outer_assessment_source_rows': rows[outer].tolist()}
        plans.append(plan)
        require(np.array_equal(np.flatnonzero(assigned == number), outer), f'Fold {number} OOF identity mismatch')
        fold_metrics = {name: recompute_metrics(target[outer], predictions[name].to_numpy()[outer]) for name in NAMES}
        directory = output / f'fold_{number}'
        fold_summary = read_json(directory / 'fold_summary.json')
        match(summary['folds'][number - 1], fold_summary, f'summary.fold{number}')
        match(fold_summary, {'fold': number, 'training_rows': len(inner), 'checkpoint_rows': len(checkpoint),
                            'assessment_rows': len(outer), 'summaries': fold_metrics, 'component_counts': COUNTS,
                            'historical_flat_weights': [.05] * 5 + [.1125] * 5 + [.1875]}, f'fold{number}')
        single = verify_checkpoint_history(directory, 'single_report.json', 'single',
            {'single_mae': 'mae_kzt', 'single_log': 'mean_absolute_log1p_error'}, 300, 45, target[inner], len(checkpoint))
        reference = verify_checkpoint_history(directory, 'reference_report.json', 'reference',
            {'reference_mae': 'mae_kzt'}, 400, 40, target[inner], len(checkpoint))
        match(fold_summary['selected_epochs'], {**single, **reference}, f'fold{number}.selected_epochs')
        selected_epochs.append(single)
        for name, folder in (('corrected', 'corrected_pool'), ('usage', 'usage_pool')):
            best = verify_pool(directory / folder, target[inner], len(checkpoint))
            match(fold_summary['pool_selected_epochs'][name], best, f'fold{number}.{name}_epochs')
        require(set(fold_summary['artifact_bytes']) == set(NAMES), f'Fold {number} artifact size recipes differ')
        for name, size in fold_summary['artifact_bytes'].items():
            require(isinstance(size, int) and size > 0, f'Fold {number} {name}: invalid transient artifact size')
        fold_results.append({'fold': number, 'summaries': fold_metrics, 'selected_epochs': {**single, **reference}})
    match(provenance['folds'], plans, 'provenance.folds')
    pooled = {name: recompute_metrics(target, predictions[name].to_numpy(dtype=float)) for name in NAMES}
    match(summary, {'status': 'development/not_independent', 'independent_evaluation': False,
                    'summaries': pooled, 'component_counts': COUNTS, 'protected_artifacts_unchanged': True}, 'summary')
    decision = independent_decision(pooled)
    match(summary['selection'], decision, 'summary.selection')
    match(read_json(output / 'selection.json'), decision, 'selection.json')
    for name in NAMES:
        sizes = [fold['artifact_bytes'][name] for fold in summary['folds']]
        match(summary['artifact_sizes_bytes'][name], {'folds': sizes, 'median': int(np.median(sizes))}, f'artifact_sizes.{name}')

    selected = decision['selected_name']
    candidate = summary['candidate_artifact']
    if selected == 'historical_recipe':
        require(candidate is None and not (output / 'development_candidate.joblib').exists()
                and not (output / 'candidate_metadata.json').exists(), 'Ineligible simplification produced a candidate artifact')
    else:
        metadata = read_json(output / 'candidate_metadata.json')
        match(candidate, metadata, 'summary.candidate_artifact')
        epoch_key = 'single_mae' if selected == 'single_forest' else selected
        epochs = int(np.rint(np.median([row[epoch_key] for row in selected_epochs])))
        artifact = local_file(output, metadata['artifact'])
        match(metadata, {'status': 'development/not_independent', 'selected_name': selected,
                         'artifact': 'development_candidate.joblib', 'artifact_sha256': sha256(artifact),
                         'artifact_bytes': artifact.stat().st_size, 'training_rows': 1787,
                         'epochs': epochs, 'estimator_count': COUNTS[selected], 'independent_evaluation': False,
                         'protocol_sha256': PROTOCOL_SHA256, 'source_sha256': SOURCE_SHA256,
                         'code_sha256': code_hashes,
                         'input_contract': {'version': 1, 'currency': 'KZT', 'snapshot_year': 2025,
                                            'price_band_inference_gate': False,
                                            'forbidden_predictors': ['price', 'price_raw', 'ad_id', 'link']}}, 'candidate_metadata')
        require(not set(metadata['input_contract']['enhanced']) & {'price', 'price_raw', 'ad_id', 'link'},
                'Forbidden candidate enhanced predictors')
        require(not set(metadata['input_contract']['full']) & {'price', 'price_raw', 'ad_id', 'link'},
                'Forbidden candidate full predictors')

    return {'status': 'passed', 'evaluation_status': 'development/not_independent',
            'original_train_rows': 1787, 'excluded_original_validation_rows': 383,
            'excluded_original_test_rows': 383, 'oof_rows': len(predictions),
            'every_original_train_row_predicted_exactly_once': True, 'source_labels_match': True,
            'outer_and_inner_fold_identities_reconstructed': True, 'fold_partitions_disjoint': True,
            'checkpoint_histories_and_target_scaler_statistics_verified': True,
            'pooled_and_fold_metrics_recomputed': True, 'all_fixed_preference_gates_recomputed': True,
            'protocol_sha256': PROTOCOL_SHA256, 'source_sha256': SOURCE_SHA256,
            'code_hashes_verified': len(code_hashes), 'protected_artifacts_verified': len(before),
            'historical_selected_model_sha256': ORIGINAL_HASHES['results/selected_model.joblib'],
            'selected_name': selected, 'selection': decision, 'recomputed_summaries': pooled,
            'recomputed_folds': fold_results,
            'limitations': ['No candidate fitting, model loading, or estimator prediction is performed.',
                            'CSV identities, recorded histories, scaler statistics and frozen source code are checked; actual runtime fit boundaries cannot be proved from saved metadata alone.',
                            'Transient fold artifact sizes are checked for internal consistency; deleted artifacts cannot be remeasured.',
                            'A refit artifact hash and declared contract are checked without deserializing or independently evaluating it.',
                            'Development OOF selection does not provide independent generalization evidence.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Completed run directory; only verification.json is written')
    args = parser.parse_args()
    source, output = args.project_source.resolve(), args.output_dir.resolve()
    require(source.is_dir() and output.is_dir(), 'Source and completed output directories must already exist')
    try:
        result = verify(source, output)
    except Exception as error:
        result = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
    result['verifier_sha256'] = sha256(Path(__file__).resolve())
    (output / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'recomputed_summaries', 'recomputed_folds', 'selection'}}, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
