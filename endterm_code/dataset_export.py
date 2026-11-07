"""Lossless source/split export with the selected model's usage proxies."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

import data
from dataset_features import EXTRA_NUMERIC, add_dataset_features
from structured_features import add_structured_features


DETAILS = {
    'price': ('KZT', 'Unchanged advertised asking price: regression target, not a verified transaction value.'),
    'price_raw': ('source currency text', 'Original target-price wording; never a predictor.'),
    'ad_id': ('source identifier', 'Original ad identity retained for provenance and split checks; never a predictor.'),
    'link': ('source URL', 'Original listing reference; no current page content is attached to this historical observation.'),
    'desc': ('raw seller text', 'Original capped seller description retained for audit; full free text is not a fit input.'),
    'year': ('model year', 'Original model year, not first-registration or observation date.'),
    'mileage_km': ('reported km', 'Original reported odometer value, including missing or suspicious values; not verified true distance.'),
    'drive': ('original parser label', 'Known unreliable original extraction retained for audit and excluded from fit inputs.'),
    'condition': ('source eligibility label', 'Original new/used category; not detailed mechanical condition.'),
    'name': ('raw seller heading', 'Original title retained for provenance; not passed directly to the estimator.'),
    'brand': ('source category', 'Whitespace-normalized original brand; unknown is not an inferred manufacturer.'),
    'model': ('source category', 'Original model name; no external catalog identity or generation is inferred.'),
    'engine_liters': ('reported litres', 'Original parsed displacement retained for comparison with engine_liters_clean.'),
    'fuel': ('normalized source category', 'Exact structured fuel wording takes priority over the earlier parser; unavailable or unsupported fuel is unknown.'),
    'transmission': ('normalized source category', 'Whitespace-normalized source category; unknown is unavailable, not guessed.'),
    'dual_fuel_claim': ('0/1 source claim', 'Supported positive gas/petrol wording; 0 means no supported claim, not confirmed absence.'),
    'fuel_text_conflict': ('0/1 source disagreement', 'Positive dual-fuel wording disagrees with structured fuel; no mechanical truth is assigned.'),
    'engine_source_conflict': ('0/1 source disagreement', 'Electric wording and positive displacement coexist; retain the literal discrepancy.'),
    'engine_not_applicable': ('0/1 source state', 'Explicit electric listing without positive displacement: corrected 0 L is inapplicability, not missingness.'),
    'engine_liters_clean': ('litres', 'Plausible source displacement (0.5–8 L), or explicit electric inapplicability 0 L; unavailable or invalid values stay missing.'),
    'engine_missing': ('0/1 missingness', 'Corrected displacement is unavailable; electric inapplicability has a separate state.'),
    'body': ('normalized source text', 'Body wording from the initial used-car description prefix; unknown remains explicit.'),
    'brand_model': ('source composite category', 'Retained brand/model categorical label; not an external catalog match.'),
    'trim': ('seller token category', 'Retained supported trim wording; not_stated does not certify equipment.'),
    'restyling_claim': ('0/1 source claim', 'Retained positive restyling wording; 0 means no supported claim.'),
    'leather_claim': ('0/1 source claim', 'Retained positive leather wording; 0 means no supported claim.'),
    'panorama_claim': ('0/1 source claim', 'Retained positive panoramic-roof wording; 0 means no supported claim.'),
    'warranty_claim': ('0/1 source claim', 'Retained positive warranty wording; current coverage is not verified.'),
    'power_kw': ('kW', 'Retained explicit description power with a recognized unit; no model-name or displacement inference.'),
    'power_missing': ('0/1 missingness', 'No supported explicit power was extracted.'),
    'log_km': ('log1p reported km', 'log1p of original mileage; missing mileage stays missing before train-fitted imputation.'),
    'mileage_missing': ('0/1 missingness', 'Original reported mileage is unavailable; no odometer value is fabricated.'),
    'vehicle_age_proxy': ('years', '2025 minus a valid positive integral model year; unavailable, nonintegral or future years stay missing. Audit only.'),
    'log_vehicle_age': ('log1p years', 'log1p(vehicle_age_proxy); unavailable or invalid model year stays missing.'),
    'log_annual_mileage_proxy': ('log1p km/year proxy', 'log1p(reported km / max(vehicle_age_proxy, 1)); invalid year or missing, negative or nonfinite mileage stays missing. Not measured annual usage.'),
    'steering': ('literal source state', 'Exact pre-КПП steering clause; not_stated is unknown, never assumed left-hand steering.'),
    'source_row_index': ('source row index', 'Original retained source index; metadata only, never a predictor.'),
    'split': ('fixed split label', 'Original seed-42 assignment; metadata only, never a predictor.'),
}


def _validate_splits(frame, splits):
    if set(splits) != {'train', 'validation', 'test'} or not frame.index.is_unique:
        raise ValueError('Split requires train/validation/test and a unique source index')
    assignments = {}
    for name, indices in splits.items():
        if not len(indices):
            raise ValueError('Each split must be nonempty')
        for index in indices:
            if index not in frame.index or index in assignments:
                raise ValueError('Split indices must cover each source row exactly once')
            assignments[index] = name
    if set(assignments) != set(frame.index):
        raise ValueError('Split indices must cover every retained source row')
    return pd.Series(assignments).reindex(frame.index)


def export_dataset(frame, splits, destination, source_path=None):
    """Export every source cell in original order with unchanged IDs/targets/split.

    The course frame has 32 loaded columns. Steering, three usage proxies and
    two split metadata fields produce a focused 38-column derived copy.
    """
    required = {'ad_id', 'price', 'desc', 'year', 'mileage_km'}
    if not required.issubset(frame):
        raise ValueError(f'Missing source columns: {sorted(required - set(frame))}')
    assignment = _validate_splits(frame, splits)
    if frame.ad_id.isna().any() or not frame.ad_id.is_unique:
        raise ValueError('Every retained ad must have a distinct nonmissing identity')
    prices = pd.to_numeric(frame.price, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('The retained source must already have finite positive prices; no target repair is performed')
    if {'steering', 'source_row_index', 'split'} & set(frame):
        raise ValueError('Reserved derived or export metadata already exists')
    enriched = add_dataset_features(add_structured_features(frame))
    pd.testing.assert_frame_equal(enriched.loc[:, frame.columns], frame)
    enriched['source_row_index'] = frame.index.to_numpy()
    enriched['split'] = assignment.to_numpy()
    fit_inputs = [column for column in [*data.FEATURES, 'steering', *EXTRA_NUMERIC] if column in enriched]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / 'used_cars_enriched.csv'
    enriched.to_csv(csv_path, index=False)
    restored = pd.read_csv(csv_path, dtype={'ad_id': str})
    if restored.ad_id.tolist() != frame.ad_id.astype(str).tolist() or not np.array_equal(restored.price.to_numpy(), prices):
        raise ValueError('Export round-trip changed retained identities or targets')

    dictionary = []
    for column in enriched:
        if column == 'price':
            role = 'target'
        elif column in {'ad_id', 'link'}:
            role = 'identifier'
        elif column in {'source_row_index', 'split'}:
            role = 'split_metadata'
        elif column == 'price_raw':
            role = 'raw_target_text'
        elif column in {'desc', 'name'}:
            role = 'raw_source_text'
        elif column == 'vehicle_age_proxy':
            role = 'derived_audit'
        elif column in {*EXTRA_NUMERIC, 'steering'}:
            role = 'derived_predictor'
        elif column in fit_inputs:
            role = 'existing_predictor'
        else:
            role = 'retained_auxiliary'
        units, meaning = DETAILS.get(column, ('source units', 'Retained without change from the loaded source frame.'))
        dictionary.append({'column': column, 'role': role, 'dtype': str(enriched[column].dtype),
                           'units': units, 'fit_input': column in fit_inputs,
                           'definition_and_missing_semantics': meaning})
    pd.DataFrame(dictionary).to_csv(destination / 'data_dictionary.csv', index=False)
    source = Path(source_path) if source_path is not None else None
    report = {
        'rows': len(enriched), 'columns': len(enriched.columns),
        'source_csv': source.name if source else None,
        'source_raw_sha256': hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
        'enriched_csv_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'retained_ad_ids_sha256': hashlib.sha256('\n'.join(frame.ad_id.astype(str)).encode()).hexdigest(),
        'preserves_ids_targets': True, 'source_row_order_preserved': True,
        'snapshot_year_for_proxies': 2025,
        'split_counts': {name: len(indices) for name, indices in splits.items()},
        'fit_input_columns': fit_inputs,
        'excluded_from_fit': [column for column in enriched if column not in fit_inputs],
        'missing_counts': {column: int(count) for column, count in enriched.isna().sum().items() if count},
        'proxy_semantics': {column: DETAILS[column][1] for column in ['vehicle_age_proxy', *EXTRA_NUMERIC, 'steering']},
        'fitted_transforms': 'None in this export. Model imputers, indicators, scalers and encoders are fitted within the training fold.',
        'source_limits': 'Historical seller wording and asking prices, not verified vehicle facts or transaction prices. No external specifications, added rows, price changes or target-based exclusions.',
    }
    (destination / 'quality_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (destination / 'README.md').write_text('''# Focused copy of the original course dataset

`used_cars_enriched.csv` preserves all 32 loaded source columns, every retained ad, original asking price, row order and historical seed-42 split. It adds literal steering, three fixed-2025 usage proxy fields and `source_row_index`/`split` metadata, for 38 columns. The original CSV is unchanged.

`vehicle_age_proxy` is 2025 minus a valid positive integral model year; it is audit only. `log_vehicle_age` is log1p(age), and `log_annual_mileage_proxy` is log1p(reported km / max(age, 1)). Invalid or unavailable years and unavailable/invalid mileage remain unknown in the relevant proxies. These are mathematical proxies, not measured time since registration or annual usage. Steering comes only from exact pre-КПП source clauses; omitted steering remains unknown.

`data_dictionary.csv` describes every field and marks the union of selected-branch fit inputs. Targets, IDs, URLs, raw free text and split metadata are excluded. Original auxiliary fields remain for audit. `quality_report.json` records provenance hashes, missing counts, split counts and proxy semantics. This export fits no transforms or models; model preprocessing is fitted inside training folds. Model metrics and the historical-evaluation caveat are documented in the project README and `results/report.json`.
''', encoding='utf-8')
    return report


def main():
    root = Path(__file__).resolve().parent
    frame, _ = data.load_data(root)
    report = export_dataset(frame, data.split_data(frame), root / 'dataset', root / 'kolesa_cars_2025.csv')
    print(json.dumps({key: report[key] for key in ['rows', 'columns', 'split_counts', 'preserves_ids_targets']}, indent=2))


if __name__ == '__main__':
    main()
