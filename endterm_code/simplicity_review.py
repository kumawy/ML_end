"""Frozen TRAIN-only development comparison; never an independent test.

No existing selected artifact is replaced. The CLI has no tuning overrides.
Private epoch limits exist solely for small synthetic behavioral tests.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ann import ANNRegressor
import data
from dataset_features import add_dataset_features
from ensemble import Component, PredictionEnsemble
from robust_ann import _PoolConfig, fit_robust_pool, make_robust_preprocessor
from structured_features import add_structured_features

NAMES = ('single_mae', 'single_log', 'single_forest', 'historical_recipe')
COUNTS = dict(zip(NAMES, (1, 1, 2, 11)))
USAGE = ('log_vehicle_age', 'log_annual_mileage_proxy')
BAND_UPPER_KZT = (3_600_000., 6_700_000., 11_500_000.)
STATUS = 'development/not_independent'


@dataclass(frozen=True)
class _ReviewConfig:
    single_epochs: int = 300
    single_patience: int = 45
    reference_epochs: int = 400
    reference_patience: int = 40
    pool_epochs: int = 300
    pool_patience: int = 45

    def __post_init__(self):
        if any(value < 1 for value in asdict(self).values()):
            raise ValueError('Epoch and patience limits must be positive')


def _json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bands(target):
    return np.searchsorted(BAND_UPPER_KZT, target, side='left')


def summarize_predictions(y, pred):
    """Pooled counts and original-KZT errors, with fixed inclusive bands."""
    target, prediction = (np.asarray(value, dtype=float).ravel() for value in (y, pred))
    if (not len(target) or target.shape != prediction.shape
            or not np.isfinite(target).all() or not np.isfinite(prediction).all()
            or (target <= 0).any() or (prediction < 0).any()):
        raise ValueError('Positive finite targets and nonnegative aligned predictions are required')
    error = np.abs(target - prediction)
    ape = error / target

    def subset(mask):
        count = int(np.sum(mask))
        hits = int(np.sum(ape[mask] <= .2))
        return {'n': count, 'mae_kzt': float(np.mean(error[mask])) if count else None,
                'within_20pct_count': hits, 'within_20pct': hits / count if count else None,
                'median_ape': float(np.median(ape[mask])) if count else None}

    return {**subset(np.ones(len(target), dtype=bool)),
            'affordable': subset(target <= BAND_UPPER_KZT[1]),
            'bands': {f'Q{i + 1}': subset(_bands(target) == i) for i in range(4)}}


def choose_candidate(summaries):
    """Apply all frozen preference gates; rank eligible recipes by simplicity."""
    if set(summaries) != set(NAMES):
        raise ValueError('Provide all four fixed candidate summaries')
    reference = summaries['historical_recipe']
    rows = []
    for name in NAMES[:-1]:
        row = summaries[name]
        if row['n'] != reference['n']:
            raise ValueError('Candidates must assess the same number of rows')
        gates = {
            'mae_within_3pct': row['mae_kzt'] <= 1.03 * reference['mae_kzt'],
            'overall_hits_not_worse': row['within_20pct_count'] >= reference['within_20pct_count'],
            'q1_hits_not_worse': row['bands']['Q1']['within_20pct_count'] >= reference['bands']['Q1']['within_20pct_count'],
            'q2_hits_not_worse': row['bands']['Q2']['within_20pct_count'] >= reference['bands']['Q2']['within_20pct_count'],
            'affordable_band_strict_hit_gain': any(row['bands'][q]['within_20pct_count'] > reference['bands'][q]['within_20pct_count'] for q in ('Q1', 'Q2')),
            'affordable_median_ape_not_worse': (row['affordable']['median_ape'] is not None
                and reference['affordable']['median_ape'] is not None
                and row['affordable']['median_ape'] <= reference['affordable']['median_ape']),
        }
        rows.append({'name': name, 'eligible': bool(all(gates.values())),
                     'estimator_count': COUNTS[name], 'gates': {k: bool(v) for k, v in gates.items()}})
    eligible = [row['name'] for row in rows if row['eligible']]
    selected = min(eligible, key=lambda name: (COUNTS[name], summaries[name]['affordable']['median_ape'],
                                             summaries[name]['mae_kzt'], name)) if eligible else 'historical_recipe'
    return {'selected_name': selected, 'eligible_names': eligible, 'candidates': rows,
            'ranking': 'estimator count, affordable median APE, global MAE, name',
            'decision': 'Refit a separate development candidate' if eligible else 'No tested simplification met the joint preference; retain historical recipe',
            'status': STATUS}


def plan_folds(training):
    """Return source-row identities for every outer and inner partition."""
    if not training.index.is_unique:
        raise ValueError('Source-row indices must be unique')
    target = training.price.to_numpy(dtype=float)
    labels, source_rows = _bands(target), training.index.to_numpy()
    result = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261107)
    for number, (develop, outer) in enumerate(splitter.split(source_rows, labels), 1):
        inner, checkpoint = train_test_split(develop, test_size=.2, stratify=labels[develop], random_state=20261108)
        result.append({'fold': number,
                       'inner_train_source_rows': source_rows[inner].astype(int).tolist(),
                       'inner_checkpoint_source_rows': source_rows[checkpoint].astype(int).tolist(),
                       'outer_assessment_source_rows': source_rows[outer].astype(int).tolist()})
    return result


def _network():
    return MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                        alpha=.001, batch_size=64, learning_rate_init=.001,
                        random_state=42, shuffle=True, early_stopping=False, max_iter=1, tol=0)


def fit_single_trajectory(training, y_train, checkpoint, y_checkpoint, destination, *, corrected=False, _config=None):
    """Fit A/B's one shared trajectory, or D's corrected reference trajectory."""
    config = _ReviewConfig() if _config is None else _config
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    y, v_y = (np.asarray(value, dtype=float).ravel() for value in (y_train, y_checkpoint))
    summarize_predictions(y, y)
    summarize_predictions(v_y, v_y)
    if len(training) != len(y) or len(checkpoint) != len(v_y):
        raise ValueError('Frames and targets must align')
    prep = (data.make_preprocessor(2, 'corrected') if corrected else
            make_robust_preprocessor(2, additional_numeric=USAGE))
    x, v = prep.fit_transform(training), prep.transform(checkpoint)
    scaler = StandardScaler().fit(np.log1p(y).reshape(-1, 1))
    z = scaler.transform(np.log1p(y).reshape(-1, 1)).ravel()
    v_z = scaler.transform(np.log1p(v_y).reshape(-1, 1)).ravel()
    criteria = ('mae_kzt',) if corrected else ('mae_kzt', 'mean_absolute_log1p_error')
    best = {criterion: (float('inf'), 0, None) for criterion in criteria}
    network, history = _network(), []
    epochs = config.reference_epochs if corrected else config.single_epochs
    patience = config.reference_patience if corrected else config.single_patience
    with threadpool_limits(limits=1):
        for epoch in range(1, epochs + 1):
            network.partial_fit(x, z)
            wrapped = ANNRegressor(network, scaler)
            prediction = wrapped.predict(v)
            values = {'mae_kzt': float(np.mean(np.abs(v_y - prediction))),
                      'mean_absolute_log1p_error': float(np.mean(np.abs(np.log1p(v_y) - np.log1p(prediction))))}
            history.append({'epoch': epoch, **values,
                'train_scaled_log_mse': float(np.mean((z - network.predict(x)) ** 2)),
                'checkpoint_scaled_log_mse': float(np.mean((v_z - network.predict(v)) ** 2)),
                **{f'improved_{key}': values[key] < best[key][0] for key in criteria}})
            for key in criteria:
                if values[key] < best[key][0]:
                    best[key] = values[key], epoch, deepcopy(network)
            if epoch - max(value[1] for value in best.values()) >= patience:
                break
    prefix = 'reference' if corrected else 'single'
    names = ('reference_mae',) if corrected else ('single_mae', 'single_log')
    models = {name: PredictionEnsemble([Component(ANNRegressor(best[key][2], scaler), 'enhanced', prep)], [1.])
              for name, key in zip(names, criteria)}
    report = {'status': STATUS, 'training_rows': len(y), 'checkpoint_rows': len(v_y),
              'epochs_run': epoch, 'max_epochs': epochs, 'patience': patience,
              'stopping_rule': 'Stop after neither selected criterion improves for patience epochs; strict comparisons keep first ties',
              'selected_epochs': {name: best[key][1] for name, key in zip(names, criteria)},
              'selected_values': {name: best[key][0] for name, key in zip(names, criteria)},
              'target_scaler': {'n_samples_seen': int(scaler.n_samples_seen_),
                                'mean_log1p': float(scaler.mean_[0]), 'scale_log1p': float(scaler.scale_[0])},
              'preprocessing_fit_split': 'inner training only', 'min_frequency': 2}
    pd.DataFrame(history).to_csv(destination / f'{prefix}_history.csv', index=False)
    _json(destination / f'{prefix}_report.json', report)
    return models, report


def _forest(training, target):
    return Pipeline([('preprocess', data.make_preprocessor(1, 'full')),
                     ('forest', RandomForestRegressor(n_estimators=200, max_features=.7,
                         min_samples_leaf=1, random_state=42, n_jobs=2))]).fit(training, target)


def _fit_pool(training, target, checkpoint, checkpoint_target, baseline, destination, usage, config):
    # The helper writes bulky checkpoint artifacts. Retain human-auditable
    # histories/reports and the selected in-memory components, then remove bulk.
    with tempfile.TemporaryDirectory(prefix='simplicity_pool_') as temporary:
        models, report = fit_robust_pool(training, target, checkpoint, checkpoint_target,
            baseline, temporary, seed=20261108, additional_numeric=USAGE if usage else (),
            _config=_PoolConfig(max_epochs=config.pool_epochs, patience=config.pool_patience))
        destination.mkdir()
        for filename in ('robust_ann_history.csv', 'robust_ann_report.json', 'robust_ann_history.png'):
            shutil.copyfile(Path(temporary) / filename, destination / filename)
        # Keep relative historical artifact labels, explicitly explain absence.
        report['checkpoint_artifact_retention'] = 'Temporary joblib checkpoints deleted; history and report retained'
        _json(destination / 'robust_ann_report.json', report)
    return models, report


def _fit_fold(features, target, fold, destination, config):
    inner, checkpoint, outer = (fold[key] for key in
        ('inner_train_source_rows', 'inner_checkpoint_source_rows', 'outer_assessment_source_rows'))
    training, check, assessment = features.loc[inner], features.loc[checkpoint], features.loc[outer]
    y, v_y = target.loc[inner].to_numpy(), target.loc[checkpoint].to_numpy()
    models, single = fit_single_trajectory(training, y, check, v_y, destination, _config=config)
    forest = _forest(training, y)
    forest_component = Component(forest, 'full')
    models['single_forest'] = PredictionEnsemble([*models['single_mae'].components, forest_component], [.75, .25])
    reference, reference_report = fit_single_trajectory(training, y, check, v_y, destination, corrected=True, _config=config)
    check_inputs = {'enhanced': check, 'full': check}
    reference_prediction = .75 * reference['reference_mae'].predict(check_inputs) + .25 * forest.predict(check)
    corrected, corrected_report = _fit_pool(training, y, check, v_y, reference_prediction,
        destination / 'corrected_pool', False, config)
    dual = corrected['ANN pool dual']
    robust = PredictionEnsemble([*dual.components, forest_component], [.75 / 5] * 5 + [.25])
    usage, usage_report = _fit_pool(training, y, check, v_y, robust.predict(check_inputs),
        destination / 'usage_pool', True, config)
    usage_mae = usage['ANN pool MAE']
    historical = PredictionEnsemble([*usage_mae.components, *dual.components, forest_component],
                                   [.05] * 5 + [.1125] * 5 + [.1875])
    models['historical_recipe'] = historical
    inputs = {'enhanced': assessment, 'full': assessment}
    predictions = {name: models[name].predict(inputs) for name in NAMES}
    artifact_bytes = {}
    with tempfile.TemporaryDirectory(prefix='simplicity_sizes_') as temporary:
        for name in NAMES:
            artifact = Path(temporary) / f'{name}.joblib'
            joblib.dump(models[name], artifact)
            artifact_bytes[name] = artifact.stat().st_size
    report = {'fold': fold['fold'], 'training_rows': len(y), 'checkpoint_rows': len(v_y),
              'assessment_rows': len(outer), 'selected_epochs': {**single['selected_epochs'], **reference_report['selected_epochs']},
              'pool_selected_epochs': {'corrected': corrected_report['member_best_epochs'],
                                       'usage': usage_report['member_best_epochs']},
              'summaries': {name: summarize_predictions(target.loc[outer], values) for name, values in predictions.items()},
              'component_counts': COUNTS, 'artifact_bytes': artifact_bytes,
              'historical_flat_weights': historical.weights,
              'artifacts': {'single_history': 'single_history.csv', 'reference_history': 'reference_history.csv',
                            'corrected_report': 'corrected_pool/robust_ann_report.json',
                            'usage_report': 'usage_pool/robust_ann_report.json'}}
    _json(destination / 'fold_summary.json', report)
    return predictions, report


def _preserved_hashes(source, excluded):
    files = set(source.glob('*.joblib'))
    for directory in ('results', 'reference'):
        files.update(path for path in (source / directory).rglob('*') if path.is_file())
    return {path.relative_to(source).as_posix(): _sha(path) for path in sorted(files)
            if excluded not in path.parents and path != excluded}


def _candidate_refit(name, features, target, epochs, output, provenance):
    prep = make_robust_preprocessor(2, additional_numeric=USAGE)
    x = prep.fit_transform(features)
    scaler = StandardScaler().fit(np.log1p(target).to_numpy().reshape(-1, 1))
    z = scaler.transform(np.log1p(target).to_numpy().reshape(-1, 1)).ravel()
    network = _network()
    with threadpool_limits(limits=1):
        for _ in range(epochs):
            network.partial_fit(x, z)
    components, weights = [Component(ANNRegressor(network, scaler), 'enhanced', prep)], [1.]
    if name == 'single_forest':
        components.append(Component(_forest(features, target), 'full'))
        weights = [.75, .25]
    model = PredictionEnsemble(components, weights)
    artifact = output / 'development_candidate.joblib'
    joblib.dump(model, artifact)
    metadata = {'status': STATUS, 'selected_name': name, 'artifact': artifact.name,
        'artifact_sha256': _sha(artifact), 'artifact_bytes': artifact.stat().st_size,
        'training_rows': len(target), 'epochs': epochs, 'estimator_count': COUNTS[name],
        'epoch_rule': 'Rounded median of five selected inner epochs; no old validation/test used',
        'independent_evaluation': False, 'protocol_sha256': provenance['protocol_sha256'],
        'source_sha256': provenance['source_sha256'], 'code_sha256': provenance['code_sha256'],
        'input_contract': {'version': 1, 'currency': 'KZT', 'snapshot_year': 2025,
            'enhanced': [*data.CORRECTED_NUMERIC, *USAGE, *data.CORRECTED_CATEGORICAL, 'steering'],
            'full': list(data.FEATURES) if name == 'single_forest' else [],
            'construction': 'For a data.engineer_vehicle_features frame: enhanced=add_structured_features(add_dataset_features(frame,snapshot_year=2025)); full=frame. Supply a dict with enhanced and, for single_forest, full.',
            'forbidden_predictors': ['price', 'price_raw', 'ad_id', 'link'],
            'price_band_inference_gate': False}}
    _json(output / 'candidate_metadata.json', metadata)
    return metadata


def run_review(project_source, output_dir):
    """Run the exact frozen protocol once in a new evidence directory."""
    source, output = Path(project_source).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing existing output directory: {output}')
    protocol = source / 'experiments/simplicity_review/PROTOCOL.md'
    frame, audit = data.load_data(source)
    split = data.split_data(frame)
    # Select TRAIN before extracting targets or deriving age/steering inputs.
    training = frame.loc[split['train']].copy().sort_index()
    if len(training) != 1787:
        raise ValueError('This frozen experiment requires exactly the original 1,787 TRAIN rows')
    target = training.price.astype(float).copy()
    plan = plan_folds(training)
    protected = _preserved_hashes(source, output)
    code_files = [*source.glob('*.py'), source / 'experiments/simplicity_review/run_simplicity_review.py']
    provenance = {'status': 'prepared_before_any_fitting', 'evaluation_status': STATUS,
        'project_source': '.', 'source_csv': 'kolesa_cars_2025.csv',
        'source_sha256': _sha(source / 'kolesa_cars_2025.csv'), 'source_git_blob_sha': audit['source_git_blob_sha'],
        'protocol_sha256': _sha(protocol), 'code_sha256': {p.relative_to(source).as_posix(): _sha(p) for p in sorted(code_files)},
        'training_rows': len(training), 'excluded_validation_rows': len(split['validation']), 'excluded_test_rows': len(split['test']),
        'train_source_rows': training.index.astype(int).tolist(), 'train_ad_ids': training.ad_id.astype(str).tolist(),
        'outer_splitter': {'n_splits': 5, 'shuffle': True, 'random_state': 20261107},
        'inner_splitter': {'name': 'train_test_split', 'test_size': .2, 'stratify': 'fixed price bands', 'random_state': 20261108},
        'band_upper_kzt': list(BAND_UPPER_KZT), 'folds': plan,
        'configuration': asdict(_ReviewConfig()), 'component_counts': COUNTS,
        'versions': {'python': platform.python_version(), **{name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'joblib')}},
        'protected_artifact_hashes': protected}
    output.mkdir(parents=True)
    shutil.copyfile(protocol, output / 'protocol_before_evaluation.md')
    _json(output / 'provenance.json', provenance)
    _json(output / 'preserved_artifacts_before.json', protected)
    oof = pd.DataFrame({'source_row': training.index.astype(int), 'ad_id': training.ad_id.astype(str),
                        'fold': 0, 'actual_price_kzt': target}, index=training.index)
    for fold in plan:
        oof.loc[fold['outer_assessment_source_rows'], 'fold'] = fold['fold']
    oof.to_csv(output / 'folds.csv', index=False)
    # Only original TRAIN rows enter feature derivation and every fit/predict.
    features = add_structured_features(add_dataset_features(training.drop(
        columns=['price', 'price_raw', 'ad_id', 'link'], errors='ignore'), snapshot_year=2025))
    fold_reports = []
    for fold in plan:
        print(f"Fitting outer fold {fold['fold']}/5 (TRAIN-only)", flush=True)
        destination = output / f"fold_{fold['fold']}"
        destination.mkdir()
        predictions, report = _fit_fold(features, target, fold, destination, _ReviewConfig())
        for name, values in predictions.items():
            oof.loc[fold['outer_assessment_source_rows'], name] = values
        fold_reports.append(report)
        # Preserve completed fold evidence even if a later fold is interrupted.
        oof.dropna(subset=list(NAMES)).to_csv(output / 'oof_predictions_partial.csv', index=False)
    oof.to_csv(output / 'oof_predictions.csv', index=False)
    (output / 'oof_predictions_partial.csv').unlink(missing_ok=True)
    summaries = {name: summarize_predictions(target, oof[name]) for name in NAMES}
    selection = choose_candidate(summaries)
    _json(output / 'selection.json', selection)
    candidate = None
    if selection['eligible_names']:
        selected = selection['selected_name']
        epoch_key = 'single_mae' if selected == 'single_forest' else selected
        epochs = int(np.rint(np.median([r['selected_epochs'][epoch_key] for r in fold_reports])))
        candidate = _candidate_refit(selected, features, target, epochs, output, provenance)
    after = {name: _sha(source / name) if (source / name).exists() else None for name in protected}
    _json(output / 'preserved_artifacts_after.json', after)
    if after != protected:
        raise RuntimeError('Protected historical artifacts changed during the experiment')
    summary = {'status': STATUS, 'independent_evaluation': False, 'summaries': summaries,
               'selection': selection, 'folds': fold_reports, 'component_counts': COUNTS,
               'artifact_sizes_bytes': {name: {'folds': [r['artifact_bytes'][name] for r in fold_reports],
                   'median': int(np.median([r['artifact_bytes'][name] for r in fold_reports]))} for name in NAMES},
               'candidate_artifact': candidate, 'protected_artifacts_unchanged': True,
               'provenance': 'provenance.json', 'predictions': 'oof_predictions.csv',
               'caveat': 'Development OOF is not independent: these data influenced historical decisions, and this OOF comparison selects the candidate. A one-hit difference does not establish statistical improvement.'}
    _json(output / 'summary.json', summary)
    lines = ['# TRAIN-only simplicity development review', '', summary['caveat'], '',
             '| Recipe | Estimators | MAE KZT | Hits ±20% | Affordable median APE | Median fold artifact bytes |',
             '|---|---:|---:|---:|---:|---:|']
    for name, values in summaries.items():
        lines.append(f"| {name} | {COUNTS[name]} | {values['mae_kzt']:.0f} | {values['within_20pct_count']}/{values['n']} | {values['affordable']['median_ape']:.4f} | {summary['artifact_sizes_bytes'][name]['median']} |")
    lines += ['', selection['decision'] + ': ' + selection['selected_name'] + '.', '',
              'The current selected_model.joblib and all protected historical artifacts are unchanged.',
              'A separate candidate, if present, is a development artifact with no independent evaluation.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary
