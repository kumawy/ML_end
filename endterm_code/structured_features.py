"""Literal steering clauses from the existing listing specifications only.

No steering value is inferred from a model, brand, price, listing ID or URL.
An omitted clause is not evidence of left-hand steering. The conservative
boundary is identical to the TRAIN-audited scratch experiment.
"""
import re

_KPP = re.compile(r'\bКПП\b', re.IGNORECASE)
_STEERING = re.compile(r'(правый|левый)\s+руль', re.IGNORECASE)


def extract_structured_steering(description):
    """Read exact comma-delimited steering clauses preceding a standalone КПП."""
    if not isinstance(description, str):
        return 'not_stated'
    boundary = _KPP.search(description)
    if boundary is None:
        return 'not_stated'
    values = set()
    for clause in description[:boundary.start()].split(','):
        match = _STEERING.fullmatch(clause.strip())
        if match:
            values.add('right_explicit' if match.group(1).casefold() == 'правый' else 'left_explicit')
    return 'conflict' if len(values) > 1 else next(iter(values), 'not_stated')


def add_structured_features(frame):
    """Return a copy with ``steering`` derived solely from the original ``desc``.

    Call this before inference with an ANN pool: each pool's enhanced
    preprocessor deliberately requires the explicit steering input column.
    """
    if 'desc' not in frame.columns:
        raise ValueError('Structured steering requires the original desc column.')
    result = frame.copy()
    result['steering'] = result['desc'].map(extract_structured_steering).astype(object)
    return result
