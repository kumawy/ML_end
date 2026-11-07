# Original-dataset feature implementation

Implemented only in staged `code/dataset_features.py`, with behavioral tests in `code/tests/test_dataset_features.py`. The original CSV, `data.py`, existing model implementations, existing split, results and saved models were not edited by this implementation task. Existing `structured_features.py` remains separate and unchanged.

## API and contract

`add_dataset_features(frame, snapshot_year=2025)` returns a new DataFrame with all original columns, values, dtypes, row order and row index retained. It requires only `desc`, `brand`, `model`, `year` and `mileage_km`. Optional original `drive` is read exclusively for its review flag. No label or identity value is read to derive a feature. Missing required columns and collisions with generated column names raise `ValueError` before modification.

The seven additional predictor candidates are explicitly exported as:

```python
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
```

`vehicle_age_proxy` is also retained for inspection but excluded from those predictor lists. The four quality columns are excluded from predictor candidates. The module does not add global collision features or make row-deletion or correction decisions.

## Source semantics

- Age is `snapshot_year - year`, with the default explicitly fixed at 2025. A missing, nonnumeric, nonintegral, nonpositive or future year produces missing age-derived values, while the raw year remains unchanged. An explicit snapshot override must be a positive integer.
- `log_vehicle_age` is `log1p(age)`. `log_annual_mileage_proxy` is `log1p(mileage_km / max(age, 1))`. Missing or invalid/negative numeric mileage makes this proxy missing without changing source mileage. This is a mathematical model-year proxy, not measured annual usage.
- Color and equipment use the exact audited source boundaries: exactly one complete comma-delimited recognized КПП clause, optional `с пробегом … км` or bare `… км`, exterior color at the next slot, then a contiguous closed equipment vocabulary. Unknown, negated, truncated and narrative clauses stop parsing. No fuzzy color standardization is applied.
- `alloy_wheels_claim` requires literal `литые диски`; `sunroof_claim` requires literal `люк`. A panoramic roof is distinct. Positive claims are `1.0`; unmentioned/unsupported claims remain NaN. A later numeric pipeline should retain training-fitted missingness indicators to distinguish unknown from a positive claim.
- Upholstery uses only literal `кожа`, `велюр`, `комбинированный` and `алькантара`. Multiple source color/material labels use `multiple_literals`, which means ambiguity for a single category and does not establish a physical contradiction. Wood is not treated as upholstery.
- Lexical pooling exactly follows the source proposal. BMW three-digit labels map to `numeric_prefix_N`; they are not renamed to externally asserted Series names. Mercedes-Benz and Lexus explicit leading alphabetic prefixes before a numeric token are brand-scoped. Unmatched source model labels remain distinct.
- Truncation is explicitly `desc.rstrip().endswith('...')` with stripped Unicode length 119–122 inclusive. TRAIN's observed markers occur at 120 or 121. This narrow format signal does not prove the upstream truncation mechanism or completeness when absent.
- Low-mileage review means valid proxy age at least 10 and known nonnegative mileage at most 1,000. Repeated-digit review requires a positive integer mileage with at least three identical digits. Original-drive review marks every nonmissing raw drive value. These are review metadata, not evidence of fraud, verified condition or permission to alter values.

## TRAIN-only verification

`check_feature_implementation.py` reproduces the audit and writes `feature_implementation_checks.json`. It selects the existing 1,787 TRAIN rows before inspecting feature coverage. Labels and identities are compared only for exact preservation; no target relationships, validation/test feature analysis or model experiment was performed in this implementation task. The common existing loader still reads the full original CSV for its preexisting eligibility rules.

All original 32 loaded TRAIN columns and the original row index are exactly equal before and after enrichment. Feature outputs are identical when labels/identity columns are removed. Every TRAIN source-field boundary and lexical-family output matches final `source_fields/proposal.py`; only the ambiguity category name is changed from the prototype's `conflict` to `multiple_literals`. There are no multiple-color TRAIN rows, so the observed color counts are identical.

| Field/check | Verified TRAIN count |
|---|---:|
| Exact known color literal | 1,281 |
| Positive alloy-wheel clause | 642 |
| Positive literal sunroof clause | 194 |
| Any upholstery material literal, including multiple labels | 321 |
| Multiple upholstery labels | 15 |
| Distinct brand-scoped lexical families | 384 |
| Description truncation signal | 1,545 |
| Low mileage on an old vehicle review | 11 |
| Repeated-digit mileage review | 43 |
| Original drive requires review | 23 |
| Missing source mileage retained | 349 |
| Missing annual-mileage proxy corresponding to source missingness | 349 |

Upholstery category counts are: `not_stated` 1,466; `велюр` 158; `кожа` 130; `multiple_literals` 15; `комбинированный` 14; `алькантара` 4. These agree with the independent final source audit.

Source CSV SHA-256 remains `b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468`. The audited proposal SHA-256 is recorded in the verification JSON.

## Tests

The 16 enrichment behavioral tests were written and observed failing before implementation because the requested module was absent. They cover source-cell and duplicate-index preservation, target/identity independence, missing mileage, fixed/overridden snapshot behavior, invalid input behavior, raw literal color labels, both mileage layouts, interior/narrative/negation/truncation boundaries, multiple source labels, complete equipment claims, lexical unmatched/cross-brand names, row-local review flags, schema validation and empty frames.

Fresh final full-project verification after the separate export module landed:

```text
python -m unittest discover -s tests -v
Ran 54 tests in 1.743s
OK
```

This comprises all 35 preexisting tests, 16 enrichment tests and the parent's 3 export tests. Before the export implementation landed, a full run showed only those 3 expected missing-export-module failures; the final full run has no failures or warnings. No end-to-end project training, accuracy search or saved-model replacement was performed here. Existing lightweight model unit tests run their own small test fixtures.
