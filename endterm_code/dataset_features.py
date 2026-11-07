"""Lossless fixed-snapshot age and mileage proxies, without target access."""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


EXTRA_NUMERIC = ['log_vehicle_age', 'log_annual_mileage_proxy']
_REQUIRED = ('year', 'mileage_km')
_GENERATED = ['vehicle_age_proxy', *EXTRA_NUMERIC]


def add_dataset_features(frame, snapshot_year=2025):
    """Return a copy with three row-local proxies from year and mileage only.

    Age is ``snapshot_year - year`` for an integral positive model year no
    later than the snapshot. Invalid or unavailable years remain unknown.
    Annual mileage is ``log1p(mileage_km / max(age, 1))`` for finite,
    nonnegative mileage; unavailable or invalid mileage remains unknown.
    The fixed default snapshot is 2025, not the current calendar year.
    These are mathematical model-year proxies, not measured vehicle usage.

    Source cells, index and row order are unchanged. Required-column and
    generated-name checks prevent silent replacement of an original field.
    """
    missing = [name for name in _REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError('Dataset enrichment requires original columns: ' + ', '.join(missing))
    collisions = [name for name in _GENERATED if name in frame.columns]
    if collisions:
        raise ValueError('Dataset enrichment would overwrite columns: ' + ', '.join(collisions))
    if isinstance(snapshot_year, bool) or not isinstance(snapshot_year, Integral) or snapshot_year <= 0:
        raise ValueError('snapshot_year must be a positive integer.')

    result = frame.copy()
    year = pd.to_numeric(frame['year'], errors='coerce').astype('float64')
    valid_year = np.isfinite(year) & year.gt(0) & year.le(snapshot_year) & year.eq(np.floor(year))
    age = (snapshot_year - year).where(valid_year)
    mileage = pd.to_numeric(frame['mileage_km'], errors='coerce').astype('float64')
    known_mileage = mileage.where(np.isfinite(mileage) & mileage.ge(0))
    result['vehicle_age_proxy'] = age
    result['log_vehicle_age'] = np.log1p(age)
    result['log_annual_mileage_proxy'] = np.log1p(known_mileage / age.clip(lower=1))
    return result
