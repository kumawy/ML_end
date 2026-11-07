"""Auditable enriched copy of the unchanged course dataset.

This module exports candidate fields and quality signals. It does not alter
the selected model's inputs, source prices, row eligibility or fixed split.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

import data
from structured_features import add_structured_features
from dataset_features import add_dataset_features, EXTRA_NUMERIC, EXTRA_CATEGORICAL, QUALITY_COLUMNS


DETAILS={
    'price':('KZT','Unchanged advertised asking price: regression target, not a verified transaction value.'),
    'price_raw':('source currency text','Original target-price wording; target information, never a predictor.'),
    'ad_id':('source identifier','Original ad identity, retained for provenance and split checks, never a predictor.'),
    'link':('source URL','Original listing reference; no current page content has been fetched or attached to an old observation.'),
    'desc':('raw seller text','Original capped/unverified description retained for audit. Full free text is not a predictor.'),
    'year':('model year','Original model year, not first-registration or observation date.'),
    'mileage_km':('reported km','Original reported odometer value retained, including missing/suspicious values; not verified true distance.'),
    'drive':('original parser label','Known unreliable original extraction; retained for audit, excluded from candidate predictors.'),
    'condition':('source eligibility label','New/used source category only; it does not measure detailed mechanical condition.'),
    'name':('raw seller heading','Original listing title retained for provenance; full title text is not passed directly to the selected estimator.'),
    'brand':('source category','Original brand text with whitespace normalization; unknown means unavailable, not an inferred manufacturer.'),
    'model':('source category','Original model name retained; no generation, engine or catalog identity is inferred from the name.'),
    'engine_liters':('reported litres','Original parsed displacement retained for comparison; use engine_liters_clean and engine-state flags for corrected modeling.'),
    'fuel':('normalized source category','Exact structured fuel wording takes priority over the earlier parser; unavailable or unsupported fuel stays unknown.'),
    'transmission':('normalized source category','Source transmission category, whitespace-normalized; unknown is unavailable rather than guessed.'),
    'dual_fuel_claim':('0/1 source claim','Supported explicit gas/petrol wording in the description; 0 means no supported positive wording, not confirmed absence.'),
    'fuel_text_conflict':('0/1 source disagreement','Positive dual-fuel wording disagrees with the retained structured fuel. No mechanical truth is assigned to either claim.'),
    'engine_source_conflict':('0/1 source disagreement','Source says electric and also supplies positive displacement; retain the literal discrepancy rather than inventing a repair.'),
    'engine_not_applicable':('0/1 source state','Explicit electric listing without positive displacement: corrected 0 L denotes inapplicability, not missing engine data.'),
    'engine_liters_clean':('litres','Retained plausible source displacement (0.5–8 L), or explicit electric inapplicability 0 L. Genuinely unavailable/invalid values stay missing.'),
    'engine_missing':('0/1 missingness','Corrected displacement is missing. Electric inapplicability is separately represented.'),
    'body':('normalized source text','Body wording extracted from the initial used-car description prefix; unknown remains explicit.'),
    'brand_model':('source composite category','Brand and original model joined as a categorical label; not an external catalog match.'),
    'trim':('seller token category','Supported advertised trim tokens only; not_stated means no supported token and does not certify equipment.'),
    'restyling_claim':('0/1 source claim','Supported positive restyling wording; 0 means no supported claim, not proof that the car is pre-restyling.'),
    'leather_claim':('0/1 source claim','Earlier conservative positive leather wording. Unlike new structured equipment fields, 0 here means no supported wording, not verified absence.'),
    'panorama_claim':('0/1 source claim','Supported positive panoramic-roof wording; 0 is no supported claim, not verified absence.'),
    'warranty_claim':('0/1 source claim','Supported positive warranty wording; no warranty validity or current coverage is verified.'),
    'power_kw':('kW','Explicit description power with a recognized unit, converted to kW. Missing values are not inferred from car names or displacement.'),
    'power_missing':('0/1 missingness','No supported explicit power value was extracted.'),
    'log_km':('log1p reported km','log1p of the original reported mileage; missing mileage remains missing before train-fitted model imputation.'),
    'mileage_missing':('0/1 missingness','Original retained mileage is unavailable; no odometer value has been fabricated.'),
    'vehicle_age_proxy':('years','2025 minus model year; model-year proxy, not time since first registration.'),
    'log_vehicle_age':('log1p years','log1p of the fixed-snapshot model-year age; invalid or unavailable year remains missing.'),
    'log_annual_mileage_proxy':('log1p km/year proxy','log1p(reported km / max(model-year age, 1)); missing mileage remains missing. This is not verified annual usage.'),
    'model_family_lexical':('literal token','Narrow brand-scoped text prefix, retaining unmatched normalized model names. No catalog equivalence, engine or generation is inferred.'),
    'color_literal':('literal source label','Exact source color clause at the allowed structured position. not_stated is unknown; multiple_literals does not establish a contradiction or verified exterior color.'),
    'alloy_wheels_claim':('1 or missing','1 means an exact positive structured seller clause. Missing means no supported claim, not confirmed absence. Numeric modeling must retain a train-fitted missingness indicator.'),
    'sunroof_claim':('1 or missing','1 means literal structured sunroof claim; a panoramic roof is distinct. Missing is unknown, not confirmed absence. Retain missingness in modeling.'),
    'upholstery_literal':('literal source label','Exact structured material wording. multiple_literals may describe mixed materials; it is not a claim of contradictory or verified condition.'),
    'description_truncation_signal':('0/1 signal','Literal trailing ellipsis near the observed source cap. Signal absent does not prove complete text.'),
    'review_low_mileage_old_vehicle':('0/1 review flag','At least ten model-year proxy years old and reported mileage at most 1000 km. Retain the value; this does not prove an error.'),
    'review_repeated_digit_mileage':('0/1 review flag','Positive integer mileage consists of one digit repeated at least three times. Retain the value; it may be genuine.'),
    'original_drive_requires_review':('0/1 review flag','Original drive value is present. The original extraction has unsupported AWD examples; no drivetrain is inferred or corrected.'),
    'steering':('literal source state','Existing strict pre-КПП exact steering clause. not_stated is unknown, never assumed left-hand steering.'),
    'source_row_index':('source row index','Original retained source index; metadata only, never a predictor.'),
    'split':('fixed split label','Original seed-42 assignment; metadata only, never a predictor.'),
}


def _validate_splits(frame,splits):
    if set(splits)!={'train','validation','test'} or not frame.index.is_unique:
        raise ValueError('Split requires train/validation/test and a unique source index')
    assignments={}
    for name,indices in splits.items():
        if not len(indices):raise ValueError('Each split must be nonempty')
        for index in indices:
            if index not in frame.index or index in assignments:
                raise ValueError('Split indices must cover each source row exactly once')
            assignments[index]=name
    if set(assignments)!=set(frame.index):
        raise ValueError('Split indices must cover every retained source row')
    return pd.Series(assignments).reindex(frame.index)


def _summary(frame):
    counts={column:int(frame[column].fillna(0).astype(bool).sum()) for column in QUALITY_COLUMNS}
    return {'rows':len(frame),'mileage_missing':int(frame.mileage_km.isna().sum()),
            'quality_flag_counts':counts,
            'literal_counts':{column:{str(key):int(value) for key,value in frame[column].value_counts(dropna=False).items()}
                              for column in ['color_literal','upholstery_literal']},
            'positive_claim_counts':{column:int(frame[column].eq(1).sum()) for column in ['alloy_wheels_claim','sunroof_claim']}}


def export_dataset(frame,splits,destination,source_path=None):
    """Export a derived copy with strict identity/target/split preservation."""
    required={'ad_id','price','desc','brand','model','year','mileage_km'}
    if not required.issubset(frame):raise ValueError(f'Missing source columns: {sorted(required-set(frame))}')
    assignment=_validate_splits(frame,splits)
    if frame.ad_id.isna().any() or not frame.ad_id.is_unique:
        raise ValueError('Every retained ad must have a distinct nonmissing identity')
    prices=pd.to_numeric(frame.price,errors='raise').to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices<=0).any():
        raise ValueError('The retained source must already have finite positive prices; no target repair is performed')
    if {'source_row_index','split'} & set(frame):raise ValueError('Reserved export metadata already exists')
    enriched=add_dataset_features(add_structured_features(frame))
    pd.testing.assert_frame_equal(enriched.loc[:,frame.columns],frame)
    enriched['source_row_index']=frame.index.to_numpy()
    enriched['split']=assignment.to_numpy()
    candidate_order=list(data.FEATURES)+['steering']+list(EXTRA_NUMERIC)+list(EXTRA_CATEGORICAL)
    candidates=list(dict.fromkeys(column for column in candidate_order if column in enriched))
    assert not set(candidates)&({'price','price_raw','ad_id','link','desc','name','source_row_index','split'}|set(QUALITY_COLUMNS))
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    csv_path=destination/'used_cars_enriched.csv'
    enriched.to_csv(csv_path,index=False)
    restored=pd.read_csv(csv_path,dtype={'ad_id':str})
    if restored.ad_id.tolist()!=frame.ad_id.astype(str).tolist() or not np.array_equal(restored.price.to_numpy(),prices):
        raise ValueError('Export round-trip changed retained identities or targets')
    dictionary=[]
    for column in enriched:
        if column=='price':role='target'
        elif column in {'ad_id','link'}:role='identifier'
        elif column in {'source_row_index','split'}:role='split_metadata'
        elif column in QUALITY_COLUMNS:role='quality_flag'
        elif column=='price_raw':role='raw_target_text'
        elif column in {'desc','name'}:role='raw_source_text'
        elif column in set(EXTRA_NUMERIC+EXTRA_CATEGORICAL):role='derived_candidate'
        elif column=='vehicle_age_proxy':role='derived_audit'
        elif column in candidates:role='existing_predictor'
        else:role='retained_auxiliary'
        units,meaning=DETAILS.get(column,('source units','Retained without change from the prior pipeline; see data.py and METHODOLOGY.md.'))
        dictionary.append({'column':column,'role':role,'dtype':str(enriched[column].dtype),'units':units,
                           'candidate_predictor':column in candidates,'definition_and_missing_semantics':meaning})
    pd.DataFrame(dictionary).to_csv(destination/'data_dictionary.csv',index=False)
    reason_columns=['review_low_mileage_old_vehicle','review_repeated_digit_mileage','original_drive_requires_review']
    reasons=[]
    for _,row in enriched.iterrows():
        items=[column for column in reason_columns if row[column]==1]
        if pd.isna(row.mileage_km):items.insert(0,'mileage_missing')
        reasons.append('|'.join(items))
    review=enriched[['ad_id','source_row_index','split','brand','model','year','mileage_km',*QUALITY_COLUMNS]].copy()
    if 'link' in enriched:review['link']=enriched.link
    review['review_reasons']=reasons
    review=review.loc[review.review_reasons.ne('')]
    review.to_csv(destination/'review_queue.csv',index=False)
    source=Path(source_path) if source_path is not None else None
    report={'rows':len(enriched),'columns':len(enriched.columns),'source_csv':source.name if source else None,
            'source_raw_sha256':hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
            'enriched_csv_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            'retained_ad_ids_sha256':hashlib.sha256('\n'.join(frame.ad_id.astype(str)).encode()).hexdigest(),
            'preserves_ids_targets':True,'source_row_order_preserved':True,'snapshot_year_for_proxies':2025,
            'split_counts':{name:len(indices) for name,indices in splits.items()},
            'candidate_feature_columns':candidates,'quality_only_columns':list(QUALITY_COLUMNS),
            'excluded_identity_target_text_metadata':[column for column in enriched if column not in candidates],
            'all_rows':_summary(enriched),'by_split':{name:_summary(enriched.loc[indices]) for name,indices in splits.items()},
            'manual_review_rows':len(review),'fitted_transforms':'None in this export. Model imputers, indicators, scalers and encoders must be fitted within the training fold.',
            'model_status':'This export does not fit or select a model. Actual field usage and model metrics are recorded in the project README and results/report.json; candidate status does not mean every extra field is enabled.',
            'source_limits':'Capped seller descriptions; unmentioned fields are unknown. Literal claims are not verified vehicle facts. No external specs, new ads, synthetic rows, price changes or target-based exclusions.'}
    (destination/'quality_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (destination/'README.md').write_text('''# Enriched copy of the original course dataset

`used_cars_enriched.csv` retains every eligible source ad, original asking price and original split. The original CSV remains unchanged. `source_row_index` and `split` are audit metadata; prices, IDs, URLs and raw free text are excluded from candidate predictors. See `data_dictionary.csv` for every field and its missing-value semantics.

The extra fields are candidate representations and review signals, not recovered ground truth. Unknown equipment remains missing; a numeric model must retain training-fitted missingness indicators. Literal upholstery/color labels and lexical families do not certify condition, catalog equivalence or specifications. Mileage/age is a fixed-2025 mathematical proxy, not measured annual usage. Suspicious mileage/drive values are retained in `review_queue.csv` for human checking; none are repaired or removed.

`quality_report.json` records row/split counts, source/export hashes and field coverage. Source-pattern discovery used TRAIN only; the fixed row-local rules are then exported for all original splits. There are no fitted transformations in this export. Actual field usage and forecast metrics are described in the project README and results/report.json. Extra columns alone are not evidence of more accurate forecasts; the selected model need not consume every exported candidate.
''',encoding='utf-8')
    return report


def main():
    root=Path(__file__).resolve().parent
    frame,_=data.load_data(root)
    report=export_dataset(frame,data.split_data(frame),root/'dataset',root/'kolesa_cars_2025.csv')
    print(json.dumps({'rows':report['rows'],'columns':report['columns'],'split_counts':report['split_counts'],
                      'manual_review_rows':report['manual_review_rows'],'preserves_ids_targets':report['preserves_ids_targets']},indent=2))


if __name__=='__main__':main()
