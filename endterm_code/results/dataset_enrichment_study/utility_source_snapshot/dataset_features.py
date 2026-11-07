"""Lossless enrichment of the original 2025 dataset, without target access.

Literal fields describe visible source wording, not verified paint, equipment,
condition, or catalog identity. Missing equipment is unknown, represented by
NaN rather than a verified negative. Quality columns are inspection metadata;
they are deliberately excluded from the extra predictor lists.
"""
from __future__ import annotations

from numbers import Integral
import re

import numpy as np
import pandas as pd


EXTRA_NUMERIC = [
    'log_vehicle_age', 'log_annual_mileage_proxy',
    'alloy_wheels_claim', 'sunroof_claim',
]
EXTRA_CATEGORICAL = [
    'model_family_lexical', 'color_literal', 'upholstery_literal',
]
QUALITY_COLUMNS = [
    'description_truncation_signal', 'review_low_mileage_old_vehicle',
    'review_repeated_digit_mileage', 'original_drive_requires_review',
]

_REQUIRED = ('desc', 'brand', 'model', 'year', 'mileage_km')
_GENERATED = ['vehicle_age_proxy'] + EXTRA_NUMERIC + EXTRA_CATEGORICAL + QUALITY_COLUMNS
_TRANSMISSION = re.compile(r'КПП\s+(?:автомат|механика|вариатор|робот)', re.I)
_MILEAGE = re.compile(r'(?:с\s+пробегом\s+)?[0-9][0-9\s\u00a0\u202f]*\s+км', re.I)

# Exact labels observed in the TRAIN source's exterior slot. Special labels
# remain literal; no fuzzy color matching or conversion to standardized hues.
_COLOR_LABELS = frozenset({
    'белый', 'черный', 'серебристый', 'серый', 'синий', 'зеленый',
    'красный', 'бордовый', 'золотистый', 'бежевый', 'голубой',
    'коричневый', 'жёлтый', 'бирюзовый', 'фиолетовый', 'оранжевый',
    'вишня', 'хамелеон', 'бронза',
})

# This closed continuation vocabulary reproduces the TRAIN-audited source
# block. Encountering anything else stops parsing before seller narrative.
_EQUIPMENT_LABELS = frozenset({
    'металлик', 'литые диски', 'тонировка', 'люк', 'панорамная крыша',
    'ветровики', 'ксенон', 'гур', 'кондиционер', 'аудиосистема', 'abs',
    'линзованная оптика', 'противотуманки', 'багажник', 'климат-контроль',
    'дневные ходовые огни', 'bluetooth', 'спойлер', 'биксенон',
    'хрустальная оптика', 'usb', 'srs', 'обвес', 'дерево', 'велюр',
    'кожа', 'комбинированный', 'mp3', 'фаркоп', 'сигнализация',
    'автозапуск', 'обогрев зеркал', 'корректор фар', 'зимний режим',
    'cd', 'рейлинги', 'алькантара', 'спортивный режим',
    'встроенный телефон', 'шторки', 'кенгурятник', 'подогрев сидений',
    'полный электропакет', 'dvd', 'круиз-контроль', 'омыватель фар',
    'сабвуфер', 'центрозамок', 'иммобилайзер', 'бесключевой доступ',
    'мультируль', 'турботаймер', 'cd-чейнджер', 'датчик слепых зон',
    'турбонаддув',
})
_UPHOLSTERY_LABELS = frozenset({'кожа', 'велюр', 'комбинированный', 'алькантара'})


def _literal_category(values):
    """Multiple visible labels are ambiguous for one category, not false facts."""
    return 'multiple_literals' if len(values) > 1 else next(iter(values), 'not_stated')


def _structured_source_fields(description):
    """Read only the contiguous block after exactly one complete КПП clause.

    An optional literal mileage clause may precede the exterior slot. Color
    must begin in that slot; a color word following equipment alone does not
    establish exterior color. An unknown, negated, truncated, or narrative
    clause ends the block. Whitespace and case are normalized for comparison.
    """
    if not isinstance(description, str):
        return 'not_stated', frozenset()
    clauses = [' '.join(part.split()).casefold() for part in description.split(',')]
    starts = [i for i, part in enumerate(clauses) if _TRANSMISSION.fullmatch(part)]
    if len(starts) != 1:
        return 'not_stated', frozenset()
    i = starts[0] + 1
    if i < len(clauses) and _MILEAGE.fullmatch(clauses[i]):
        i += 1
    colors = set()
    if i < len(clauses) and clauses[i] in _COLOR_LABELS:
        colors.add(clauses[i])
        i += 1
    equipment = set()
    while i < len(clauses):
        clause = clauses[i]
        if clause in _EQUIPMENT_LABELS:
            equipment.add(clause)
        elif colors and clause in _COLOR_LABELS:
            colors.add(clause)
        else:
            break
        i += 1
    return _literal_category(colors), frozenset(equipment)


def _lexical_model_family(brand, model):
    """Group a few literal prefixes; retain every unmatched brand/model label."""
    if not isinstance(brand, str) or not isinstance(model, str):
        return 'unknown'
    brand, model = brand.strip(), model.strip()
    if not brand or not model:
        return 'unknown'
    prefix = None
    if brand == 'BMW':
        match = re.fullmatch(r'([0-9])[0-9]{2}', model)
        if match:
            prefix = 'numeric_prefix_' + match.group(1)
    elif brand in {'Mercedes-Benz', 'Lexus'}:
        match = re.match(r'^([A-Za-z]+)\s+[0-9]', model)
        if match:
            prefix = 'literal_prefix_' + match.group(1)
    return brand + ' | ' + (prefix if prefix else 'original_model:' + model)


def _truncation_signal(description):
    """Final ASCII '...' at 119–122 Unicode characters, after trailing spaces.

    TRAIN observed the marker only at lengths 120 or 121. The explicit narrow
    119–122 window is a review signal for the source snippet format, not proof
    of the upstream truncation process or a general ellipsis detector.
    """
    if not isinstance(description, str):
        return False
    snippet = description.rstrip()
    return 119 <= len(snippet) <= 122 and snippet.endswith('...')


def _repeated_digit_mileage(value):
    if pd.isna(value) or value <= 0 or not np.isfinite(value) or value != np.floor(value):
        return False
    digits = str(int(value))
    return len(digits) >= 3 and len(set(digits)) == 1


def add_dataset_features(frame, snapshot_year=2025):
    """Return a copy with twelve source-derived fields; never alter source cells.

    Requires only desc, brand, model, year, and mileage_km. The optional raw
    drive column is read solely to mark its nonmissing values for review.
    Missing required columns or generated-column name collisions raise
    ValueError, preventing silent replacement of a source column.

    ``vehicle_age_proxy`` is snapshot_year minus an integral positive model
    year no later than the snapshot. Missing, nonnumeric, nonintegral, zero,
    negative, or future years produce NaN age-derived values. The explicit
    snapshot argument must be a positive integer; the default is fixed at
    2025 and never reads the wall clock. ``log_annual_mileage_proxy`` divides
    finite nonnegative mileage by max(age, 1), then applies log1p; it is not
    certified annual usage. Missing/invalid mileage stays unknown in the
    proxy, while the original mileage column remains untouched.

    Equipment claims are 1.0 only for exact complete positive clauses in the
    source block and NaN otherwise. Sunroof requires literal ``люк``; a
    panoramic-roof clause is distinct. Categorical ``multiple_literals``
    means multiple source labels, not a verified physical contradiction.

    The four quality columns are row-local review signals, not predictions
    or corrections. An unavailable raw drive column yields False throughout;
    any nonmissing original drive value (including a literal empty string)
    requires review because the historical extraction was unreliable.
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

    parsed = frame['desc'].map(_structured_source_fields)
    result['model_family_lexical'] = [
        _lexical_model_family(brand, model) for brand, model in zip(frame['brand'], frame['model'])
    ]
    result['model_family_lexical'] = result['model_family_lexical'].astype(object)
    result['color_literal'] = parsed.map(lambda fields: fields[0]).astype(object)
    equipment = parsed.map(lambda fields: fields[1])
    result['alloy_wheels_claim'] = equipment.map(
        lambda labels: 1.0 if 'литые диски' in labels else np.nan).astype('float64')
    result['sunroof_claim'] = equipment.map(
        lambda labels: 1.0 if 'люк' in labels else np.nan).astype('float64')
    result['upholstery_literal'] = equipment.map(
        lambda labels: _literal_category(labels & _UPHOLSTERY_LABELS)).astype(object)

    result['description_truncation_signal'] = frame['desc'].map(_truncation_signal).astype(bool)
    result['review_low_mileage_old_vehicle'] = (age.ge(10) & known_mileage.le(1000)).astype(bool)
    result['review_repeated_digit_mileage'] = mileage.map(_repeated_digit_mileage).astype(bool)
    result['original_drive_requires_review'] = frame['drive'].notna() if 'drive' in frame.columns else False
    return result
