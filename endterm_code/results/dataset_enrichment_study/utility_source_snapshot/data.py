"""Unchanged source/split with explicit, conservative vehicle-text features.

All extraction is per row and uses the existing specifications/descriptions only.
Learned imputation, scaling and category grouping remain inside model pipelines.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUMERIC = [
    'year', 'log_km', 'mileage_missing', 'engine_liters_clean',
    'engine_missing', 'engine_not_applicable', 'engine_source_conflict',
    'dual_fuel_claim', 'fuel_text_conflict', 'restyling_claim',
    'leather_claim', 'panorama_claim', 'warranty_claim', 'power_kw', 'power_missing',
]
CATEGORICAL = ['brand', 'model', 'body', 'fuel', 'transmission', 'brand_model', 'trim']
FEATURES = NUMERIC + CATEGORICAL
CORRECTED_NUMERIC = [
    'year', 'log_km', 'mileage_missing', 'engine_liters_clean',
    'engine_missing', 'engine_not_applicable', 'engine_source_conflict',
]
CORRECTED_CATEGORICAL = ['brand', 'model', 'body', 'fuel', 'transmission']
SOURCE = 'https://www.kaggle.com/datasets/mutashevdarkhan/kolesa-cars-2025'

_FUEL_NAMES = {
    'электричество': 'electric', 'газ-бензин': 'gas_petrol',
    'гибрид': 'hybrid', 'дизель': 'diesel', 'бензин': 'petrol', 'газ': 'gas',
}
_FUEL_SPEC = re.compile(
    r'(?:^|,)\s*(электричество|газ\s*[-–—]\s*бензин|гибрид|дизель|бензин|газ)\s*(?=,|$)',
    re.IGNORECASE,
)
_DUAL_FUEL = re.compile(r'\bгаз\s*[-–—]\s*бензин\b', re.IGNORECASE)
_ENGINE_SPEC = re.compile(r'(?<![\w.,])([0-9]+(?:[.,][0-9]+)?)\s*л\b', re.IGNORECASE)
_POWER = re.compile(
    r'(?<![\w.,])([0-9]+(?:[.,][0-9]+)?)\s*(квт|kw|л\s*\.?\s*с\.?|hp|bhp)\b',
    re.IGNORECASE,
)
_CLAIMS = {
    'restyling_claim': re.compile(r'\b(?:рестайлинг|рестайлинг[а-я]+|restyling|facelift)\b', re.IGNORECASE),
    'leather_claim': re.compile(r'\b(?:кожа|кожи|кожан(?:ый|ая|ое)\s+салон|кожаный|салон\s+из\s+кожи|leather\s+(?:seats|interior))\b', re.IGNORECASE),
    'panorama_claim': re.compile(r'\b(?:панорам(?:ная\s+крыша|ный\s+люк|а|ы)|panoramic\s+(?:roof|sunroof))\b', re.IGNORECASE),
    'warranty_claim': re.compile(r'\b(?:на\s+гарантии|стоит\s+гарантии|гарантия\s+до|under\s+warranty)\b', re.IGNORECASE),
}
# Token boundaries prevent model-name substrings from inventing a trim. This
# deliberately short vocabulary records an advertised token, not certified trim.
_TRIMS = {
    'AMG': re.compile(r'\bamg\b(?!\s+(?:garage|service|сервис|гараж)\b)', re.IGNORECASE),
    'GTS': re.compile(r'\bgts\b', re.IGNORECASE),
    'GTI': re.compile(r'\bgti\b', re.IGNORECASE),
    'M Sport': re.compile(r'\bm\s*[- ]?\s*sport\b', re.IGNORECASE),
    'F Sport': re.compile(r'\bf\s*[- ]?\s*sport\b', re.IGNORECASE),
    'GR Sport': re.compile(r'\bgr\s*[- ]?\s*sport\b', re.IGNORECASE),
}


def _affirmed(text, match):
    """Reject local negation/absence; a zero means no supported positive claim."""
    before = re.split(r'[,;.!?\n]', text[:match.start()])[-1][-55:]
    after = re.split(r'[,;.!?\n]', text[match.end():])[0][:55]
    negation_before = re.search(
        r'\b(?:без|не|нет|отсутствует|отсутствуют|no|not|without)\b(?:\s+[\w-]+){0,3}\s*$',
        before, re.IGNORECASE,
    )
    negation_after = re.match(
        r'\s*(?:(?:уже|больше|сейчас)\s+)?(?:нет|не\s+имеется|отсутствует|отсутствуют|'
        r'закончилась|истекла|отсутствует|ended|expired)\b', after, re.IGNORECASE,
    )
    return not (negation_before or negation_after)


def _has_claim(text, pattern):
    return int(any(_affirmed(text, match) for match in pattern.finditer(text)))


def _explicit_power_kw(text):
    """Only an explicit power unit qualifies; never parse price or bare numbers."""
    values = []
    for match in _POWER.finditer(text):
        if not _affirmed(text, match):
            continue
        value = float(match.group(1).replace(',', '.'))
        unit = re.sub(r'[\s.]', '', match.group(2).lower())
        factor = 1 if unit in ('квт', 'kw') else (0.73549875 if unit == 'лс' else 0.745699872)
        power = value * factor
        if 10 <= power <= 1500:
            values.append(power)
    # Multiple disagreeing explicit claims are ambiguous rather than a license
    # to select the largest (or the asking-price-looking) number.
    return values[0] if values and np.allclose(values, values[0], rtol=.02) else np.nan


def engineer_vehicle_features(frame):
    """Derive predictors without reading price, listing ID or URL.

    The prefix before КПП is the source's structured specification. Narrative
    dual-fuel claims are separately exposed when this specification says petrol.
    This function also supports feature generation for later individual records.
    """
    used = frame.copy()
    descriptions = used.desc.fillna('').astype(str)
    specs = descriptions.str.split(r'\bКПП\b', n=1, regex=True).str[0]
    fuel_specs = specs.map(lambda text: _FUEL_SPEC.search(text))
    explicit_fuel = fuel_specs.map(
        lambda match: _FUEL_NAMES[re.sub(r'\s*[-–—]\s*', '-', match.group(1).lower())]
        if match else None,
    )
    source_fuel = used.fuel.astype('string').str.strip().str.lower()
    allowed_fuels = set(_FUEL_NAMES.values())
    source_fuel = source_fuel.where(source_fuel.isin(allowed_fuels))
    used['fuel'] = explicit_fuel.combine_first(source_fuel).fillna('unknown')
    used['dual_fuel_claim'] = descriptions.map(lambda text: _has_claim(text, _DUAL_FUEL))
    # Parser mistakes fixed by a literal structured spec are counted in the audit;
    # this flag represents a genuine source-text disagreement, not a parser bug.
    used['fuel_text_conflict'] = (used.dual_fuel_claim.eq(1) & ~used.fuel.eq('gas_petrol')).astype(int)

    recovered = specs.map(lambda text: _ENGINE_SPEC.search(text)).map(
        lambda match: float(match.group(1).replace(',', '.')) if match else np.nan,
    )
    source_engine = pd.to_numeric(used.engine_liters, errors='coerce')
    reported_engine = source_engine.combine_first(recovered)
    has_positive_engine = source_engine.gt(0) | recovered.gt(0)
    electric = used.fuel.eq('electric')
    used['engine_source_conflict'] = (electric & has_positive_engine).astype(int)
    used['engine_not_applicable'] = (electric & ~has_positive_engine).astype(int)
    used['engine_liters_clean'] = reported_engine.where(reported_engine.between(.5, 8))
    used.loc[used.engine_not_applicable.eq(1), 'engine_liters_clean'] = 0.0
    used['engine_missing'] = used.engine_liters_clean.isna().astype(int)

    used['body'] = descriptions.str.extract(r'Б/у\s+([^,]+)', expand=False).str.strip().str.lower()
    for col in ['brand', 'model', 'body', 'transmission']:
        normalized = used[col].astype('string').str.strip()
        used[col] = normalized.mask(normalized.eq('')).fillna('unknown').astype(object)
    used['brand_model'] = used.brand + ' | ' + used.model
    # Use both the listing title and description, excluding service-shop uses.
    trim_text = used['name'].fillna('').astype(str) + ', ' + descriptions
    used['trim'] = trim_text.map(
        lambda text: '+'.join(name for name, pattern in _TRIMS.items() if _has_claim(text, pattern)) or 'not_stated',
    )
    for name, pattern in _CLAIMS.items():
        used[name] = descriptions.map(lambda text: _has_claim(text, pattern))
    used['power_kw'] = descriptions.map(_explicit_power_kw)
    used['power_missing'] = used.power_kw.isna().astype(int)
    used['log_km'] = np.log1p(used.mileage_km)
    used['mileage_missing'] = used.mileage_km.isna().astype(int)
    used[CATEGORICAL] = used[CATEGORICAL].astype(object)
    used[NUMERIC] = used[NUMERIC].astype('float64')
    return used


def load_data(root: Path):
    """Apply deterministic per-row cleaning; no fitted preprocessing or target selection."""
    path = Path(root) / 'kolesa_cars_2025.csv'
    content = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    if blob != '01e57295927f91ffd7996745ce6a867b6d41b0f7':
        raise ValueError('CSV differs from the original midterm dataset; verify its provenance.')
    raw = pd.read_csv(path)
    expected = {'name','price','price_raw','desc','link','brand','model','condition','year','mileage_km','engine_liters','fuel','transmission','drive'}
    if not expected.issubset(raw.columns):
        raise ValueError(f'Missing columns: {expected - set(raw.columns)}')
    if not raw.price_raw.astype('string').str.contains('₸', regex=False).all():
        raise ValueError('Source prices are not all labelled KZT')
    raw['ad_id'] = raw.link.astype('string').str.extract(r'/show/([0-9]+)', expand=False)
    if raw.ad_id.isna().any():
        raise ValueError('Source listing IDs missing')
    used = raw.loc[raw.condition.eq('used')].copy()
    on_order = used.desc.astype('string').str.contains('На заказ', case=False, na=False)
    before_dedup = used.loc[~on_order].copy()
    used = before_dedup.drop_duplicates('ad_id').copy()
    valid = used.price.gt(0) & used.year.between(1950,2025) & (used.mileage_km.isna() | used.mileage_km.ge(0)) & used.brand.notna()
    invalid = int((~valid).sum())
    used = used.loc[valid].copy()
    original_fuel = used.fuel.copy()
    used = engineer_vehicle_features(used)
    audit = {
        'source_url': SOURCE, 'source_git_blob_sha': blob,
        'source_sha256': hashlib.sha256(content).hexdigest(),
        'raw_rows': len(raw), 'raw_columns': 14,
        'new_rows_removed': int(raw.condition.eq('new').sum()),
        'on_order_rows_removed': int(on_order.sum()),
        'duplicate_used_rows_removed': int(before_dedup.ad_id.duplicated().sum()),
        'invalid_rows_removed': invalid, 'clean_rows': len(used),
        'missing_mileage': int(used.mileage_km.isna().sum()),
        'missing_engine': int(used.engine_liters_clean.isna().sum()),
        'fuel_from_structured_electric': int(used.fuel.eq('electric').sum()),
        'fuel_from_structured_gas_petrol': int(used.fuel.eq('gas_petrol').sum()),
        'source_fuel_missing_recovered': int((original_fuel.isna() & used.fuel.ne('unknown')).sum()),
        'source_petrol_relabelled_gas_petrol': int((original_fuel.eq('petrol') & used.fuel.eq('gas_petrol')).sum()),
        'narrative_dual_fuel_claims': int(used.dual_fuel_claim.sum()),
        'fuel_text_conflicts': int(used.fuel_text_conflict.sum()),
        'unknown_fuel': int(used.fuel.eq('unknown').sum()),
        'engine_structurally_not_applicable': int(used.engine_not_applicable.sum()),
        'engine_source_conflicts': int(used.engine_source_conflict.sum()),
        'fuel_counts': {str(k): int(v) for k, v in used.fuel.value_counts().items()},
        'explicit_claim_counts': {name: int(used[name].sum()) for name in _CLAIMS},
        'explicit_trim_counts': {str(k): int(v) for k, v in used.trim.value_counts().items()},
        'explicit_power_rows': int(used.power_kw.notna().sum()),
        'feature_semantics': {
            'fuel': 'Literal structured pre-КПП fuel takes precedence; unknown is explicit. Five narrative dual-fuel/structured-petrol disagreements (including one spaced em dash) retain petrol and receive dual_fuel_claim/fuel_text_conflict flags.',
            'engine': 'Electric without a positive literal displacement is 0 L plus engine_not_applicable=1. Electric with positive reported displacement retains the reported valid value and gets engine_source_conflict=1. Invalid/missing combustion displacement stays NaN plus engine_missing=1.',
            'claims': 'Positive text claims only, with local negation rejected; zero means not positively stated, not verified absence. Trim tokens describe seller wording, not authenticated trim.',
            'power': 'Explicit unit required: Russian л.с./лс uses metric horsepower ×0.73549875; hp/bhp uses ×0.745699872; kW unchanged. Conflicting values remain missing; no inferred model specifications.',
            'inputs': 'Only existing title/specification/description and original vehicle columns; no price, price_raw, listing ID or URL predictors.',
        },
        'source_missing_counts': {k:int(v) for k,v in raw.isna().sum().items()},
        'target': 'advertised asking price of used cars', 'currency':'KZT',
        'snapshot': '2025; individual listing dates unavailable',
    }
    return used, audit


def split_data(frame):
    """Reproduce midterm split exactly, including original source row order."""
    develop, test = train_test_split(frame.index, test_size=.15, random_state=SEED)
    train, validation = train_test_split(develop, test_size=.15/.85, random_state=SEED)
    result = {'train':train, 'validation':validation, 'test':test}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if set(frame.loc[result[a],'ad_id']) & set(frame.loc[result[b],'ad_id']):
            raise ValueError('Listing leakage between splits')
    return result


def make_preprocessor(min_frequency=5, feature_set='full'):
    """Fresh preprocessing, fitted inside each CV training fold only.

    ``corrected`` uses the original vehicle predictors plus explicit engine
    states on the same corrected-fuel frame; ``full`` retains all text features.
    Extra input columns are dropped by ColumnTransformer in either mode.
    """
    if feature_set == 'full':
        numeric, categorical = NUMERIC, CATEGORICAL
    elif feature_set == 'corrected':
        numeric, categorical = CORRECTED_NUMERIC, CORRECTED_CATEGORICAL
    else:
        raise ValueError("feature_set must be 'full' or 'corrected'")
    return ColumnTransformer([
        ('num', Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]), numeric),
        ('cat', Pipeline([('impute',SimpleImputer(strategy='constant', fill_value='unknown')),('encode',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=min_frequency,sparse_output=False))]), categorical),
    ], sparse_threshold=0)
