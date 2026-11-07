"""Original midterm cleaning and leakage-safe preprocessing for the endterm."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUMERIC = ['year', 'log_km', 'mileage_missing', 'engine_liters_clean']
CATEGORICAL = ['brand', 'model', 'body', 'fuel', 'transmission']
FEATURES = NUMERIC + CATEGORICAL
SOURCE = 'https://www.kaggle.com/datasets/mutashevdarkhan/kolesa-cars-2025'


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
    recovered = pd.to_numeric(used.desc.astype('string').str.extract(r'([0-9]+(?:[.,][0-9]+)?)\s*л',expand=False).str.replace(',','.',regex=False),errors='coerce')
    used['engine_liters_clean'] = used.engine_liters.combine_first(recovered)
    used.loc[~used.engine_liters_clean.between(.5,8), 'engine_liters_clean'] = np.nan
    used['body'] = used.desc.astype('string').str.extract(r'Б/у\s+([^,]+)',expand=False).str.strip().str.lower()
    for col in CATEGORICAL:
        used[col] = used[col].astype('string').str.strip().astype(object)
        used[col] = used[col].where(pd.notna(used[col]), np.nan)
    used['log_km'] = np.log1p(used.mileage_km)
    used['mileage_missing'] = used.mileage_km.isna().astype(int)
    # Nullable pandas Float64 from text extraction must become numpy floats.
    used[NUMERIC] = used[NUMERIC].astype('float64')
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
        'source_missing_counts': {k:int(v) for k,v in raw.isna().sum().items()},
        'target': 'advertised asking price of used cars', 'currency':'KZT',
        'snapshot': '2025; individual listing dates unavailable',
    }
    return used, audit


def make_preprocessor():
    """A fresh estimator per model; fit inside each CV training fold only."""
    return ColumnTransformer([
        ('num', Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]), NUMERIC),
        ('cat', Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=5,sparse_output=False))]), CATEGORICAL),
    ], sparse_threshold=0)
