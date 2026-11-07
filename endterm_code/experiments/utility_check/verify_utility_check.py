"""Independently check saved utility artifacts without fitting or using holdout labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    protocol = json.loads((output / 'protocol_before_evaluation.json').read_text())
    report = json.loads((output / 'report.json').read_text())
    provenance = json.loads((output / 'provenance.json').read_text())
    source = (output / provenance['project_source_relative_to_output']).resolve()
    roles = pd.read_csv(output / 'fold_roles.csv', dtype={'ad_id': str})
    predictions = pd.read_csv(output / 'oof_predictions.csv', dtype={'ad_id': str})
    histories = pd.read_csv(output / 'fold_histories.csv')
    folds = pd.read_csv(output / 'fold_metrics.csv')
    aggregate = pd.read_csv(output / 'aggregate_metrics.csv')
    selected = pd.read_csv(output / 'selected_epochs.csv')
    manifest = pd.read_csv(source / 'results' / 'split_manifest.csv', dtype={'ad_id': str})
    train = manifest.loc[manifest['split'].eq('train')].reset_index(drop=True)
    names = [item['name'] for item in protocol['representations_in_fixed_order']]
    assert len(train) == 1787 and len(names) == 5
    assert aggregate.representation.tolist() == names
    assert folds.shape[0] == selected.shape[0] == report['fits_completed'] == 15
    assert report['status'] == 'complete'
    assert len(predictions) == 5 * len(train)
    assert set(roles.source_row) == set(train.source_row)
    assert not set(predictions.source_row) & set(manifest.loc[~manifest['split'].eq('train'), 'source_row'])
    identity = train.loc[:, ['source_row', 'ad_id']].copy()
    identity.insert(0, 'train_position', np.arange(len(train)))
    expected_roles = []
    for number, (outer_train, outer_holdout) in enumerate(
            KFold(n_splits=3, shuffle=True, random_state=42).split(train), 1):
        inner_train, checkpoint = next(ShuffleSplit(
            n_splits=1, test_size=.20, random_state=20261107).split(outer_train))
        for name, positions in [('inner_train', outer_train[inner_train]),
                                ('inner_checkpoint', outer_train[checkpoint]),
                                ('outer_holdout', outer_holdout)]:
            saved = roles.loc[roles.fold.eq(number) & roles.role.eq(name)]
            pd.testing.assert_frame_equal(saved.loc[:, identity.columns].reset_index(drop=True),
                                          identity.iloc[positions].reset_index(drop=True))
            expected_roles.append((number, name, len(positions)))
    for name in names:
        own = predictions.loc[predictions.representation.eq(name)]
        assert not own.train_position.duplicated().any()
        assert set(own.train_position) == set(range(len(train)))
        assert np.isfinite(own[['actual_price_kzt', 'predicted_price_kzt']].to_numpy()).all()
        assert own[['actual_price_kzt', 'predicted_price_kzt']].gt(0).all().all()
        for number in (1, 2, 3):
            predicted = own.loc[own.fold.eq(number)]
            saved = roles.loc[roles.fold.eq(number) & roles.role.eq('outer_holdout')]
            pd.testing.assert_frame_equal(predicted.loc[:, identity.columns].reset_index(drop=True),
                                          saved.loc[:, identity.columns].reset_index(drop=True))
            history = histories.loc[histories.representation.eq(name) & histories.fold.eq(number)]
            fold = folds.loc[folds.representation.eq(name) & folds.fold.eq(number)].iloc[0]
            assert history.epoch.tolist() == list(range(1, len(history) + 1))
            assert len(history) <= 150 and len(history) == fold.epochs_run
            minimum = history.loc[history.checkpoint_mae_kzt.idxmin()]
            assert fold.selected_epoch == minimum.epoch
            assert history.loc[history.is_selected_epoch, 'epoch'].tolist() == [fold.selected_epoch]
            assert fold.target_scaler_n_samples_seen == fold.inner_train_rows
            assert fold.inner_checkpoint_rows == 239
            assert len(predicted) == fold.outer_holdout_rows
            if fold.stop_reason == 'patience':
                assert fold.epochs_run - fold.selected_epoch == 25
            else:
                assert fold.epochs_run == 150
            relative = np.abs(predicted.actual_price_kzt - predicted.predicted_price_kzt) / predicted.actual_price_kzt
            expected = [np.mean(np.abs(predicted.actual_price_kzt - predicted.predicted_price_kzt)),
                        np.mean(relative <= .20), np.median(relative)]
            np.testing.assert_allclose(fold[['mae_kzt', 'within_20pct', 'median_ape']].to_numpy(dtype=float),
                                       expected, rtol=1e-12, atol=1e-9)
        relative = np.abs(own.actual_price_kzt - own.predicted_price_kzt) / own.actual_price_kzt
        expected = [np.mean(np.abs(own.actual_price_kzt - own.predicted_price_kzt)),
                    np.mean(relative <= .20), np.median(relative)]
        row = aggregate.loc[aggregate.representation.eq(name)].iloc[0]
        np.testing.assert_allclose(row[['mae_kzt', 'within_20pct', 'median_ape']].to_numpy(dtype=float),
                                   expected, rtol=1e-12, atol=1e-9)
    assert set(selected.representation) == set(names)
    assert protocol['maximum_fits'] == 15 and protocol['maximum_epochs'] == 150 and protocol['patience'] == 25
    assert provenance['validation_test_targets_extracted'] is False
    assert provenance['validation_test_predictions_made'] is False
    assert provenance['new_price_based_filtering'] is False
    hash_checks = {}
    for name, expected in provenance['source_sha256'].items():
        current = hashlib.sha256((source / name).read_bytes()).hexdigest()
        hash_checks[name] = current == expected
    assert all(hash_checks.values()), hash_checks
    assert hashlib.sha256((Path(__file__).resolve().parent / 'run_utility_check.py').read_bytes()).hexdigest() == provenance['script_sha256']
    assert hashlib.sha256((output / 'protocol_before_evaluation.json').read_bytes()).hexdigest() == report['protocol_sha256']
    assert not [file for file in output.rglob('*') if file.suffix in {'.pkl', '.joblib'}]
    checks = {
        'status': 'passed', 'fits': 15, 'representations': 5, 'original_train_rows': len(train),
        'oof_rows': len(predictions), 'fold_history_rows': len(histories),
        'fold_roles_match_predeclared_sklearn_splits': True,
        'every_train_row_predicted_once_per_representation': True,
        'validation_test_rows_predicted': 0,
        'checkpoint_selection_matches_inner_mae_minimum': True,
        'all_positive_finite_predictions': True,
        'all_aggregate_and_fold_metrics_recomputed': True,
        'source_hash_checks': hash_checks,
        'all_fits_at_or_below_150_epochs_with_25_patience': True,
        'no_model_saved': True,
    }
    (output / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
