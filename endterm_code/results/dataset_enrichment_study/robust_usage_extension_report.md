# Explicit age/usage ANN integration

Changed only product `code/robust_ann.py` and its meaningful behavioral tests in `code/tests/test_robust_ann.py`. Main-entrypoint, export, source dataset, `dataset_features.py`, base ANN, current model results, and documentation integration remain owned by the parent workflow.

## API

```python
make_robust_preprocessor(min_frequency=2, *, additional_numeric=())

fit_robust_pool(
    train_frame, y_train, validation_frame, y_val,
    baseline_predictions, output_dir, seed=42, *,
    additional_numeric=(), _config=None,
)

select_dual_blend(
    validation_predictions, y_val, baseline_name, ann_names,
    forest_name, weights=(.25, .5, .75), *,
    include_baseline_blends=False,
)
```

The only allowed additional numeric labels are distinct `log_vehicle_age` and `log_annual_mileage_proxy` entries. Target, identity, already-present base features, unknown names, duplicates and malformed declarations are rejected before fitting or creating model-output directories. The caller must provide requested usage columns, created by `dataset_features.add_dataset_features`; the training helper continues to add literal steering on a copy. Both training and validation require all locally declared inputs.

Additional numeric inputs join the original numeric list in the same median-imputation and scaling pipeline. No new missingness indicator or transformer is added. Base feature globals are unchanged. Reports and history identify the actual local feature set; reports include actual `numeric_inputs` and explicit `additional_numeric_inputs`. Default pool names stay `ANN pool MAE` and `ANN pool dual`.

The default selector still has nine candidates in the original declaration order. Explicit `include_baseline_blends=True` adds each ANN pool blended with the original baseline at all declared weights, for 15 candidates with the default grid. Eligibility and normalized ranking continue to reference the original baseline: strictly lower validation MAE and strictly higher within-20% rate, then minimum equal-weight normalized MAE and miss fraction. Exact score ties retain the first declared candidate.

## Red/green tests

Four new behavioral tests were added before implementation and observed failing for the absent optional APIs. They then passed after the implementation:

- Combined numeric preprocessing learns medians/scales exclusively from training. Opted-in values change transformed inputs; the default transformer ignores those same undeclared columns. No module-global feature contract changes.
- The allowlist rejects target/identity/unknown/duplicate declarations before fitting or creating a model-output directory.
- The training helper requires opted-in columns in both input frames, records the real feature contract, retains default pool names and supports exact serialized-prediction round trips.
- A hand-computable fixture uniquely selects a pool/baseline mixture unavailable in the old grid, while the default still selects its original pool/forest mixture. Scores are independently derived, with the original baseline normalization preserved.

All 58 full-project tests pass, including all previous 54 tests. The default test path remains unchanged and the original checkpoint/serialization tests still pass.

## Exact frozen-research reproduction

`verify_robust_usage_extension.py` reproduces the saved five-seed usage pools using the explicit product kwargs, with no monkeypatching and a separate `robust_usage_equivalence/pool/` output directory. It only predicts the existing validation rows; no test prediction or new model search is performed.

The verification results in `robust_usage_equivalence/verification.json` establish:

| Check | Result |
|---|---|
| New default preprocessor vs saved original validation transform | Exactly equal |
| Product MAE usage pool vs frozen research validation predictions | Maximum absolute deviation 0.0 KZT |
| Product dual usage pool vs frozen research validation predictions | Maximum absolute deviation 0.0 KZT |
| All 15 candidate MAE/hit/dual scores vs frozen research | Maximum deviation 0.0 |
| Encoded usage input count | 309 |
| Module feature globals | Unchanged |
| Selected components | 0.25 usage MAE pool + 0.75 prior baseline |
| Selected validation MAE | 2,238,371.607965701 KZT |
| Selected validation within ±20% | 259 / 383 = 0.6762402088772846 |

Source CSV SHA-256 remains `b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468`. The explicit-kwargs reproduction took 5.54 seconds in the provided environment. Defaults retain the prior numeric order, preprocessing, model names, network/training algorithm, checkpoint calculations and nine-candidate blend grid; the default preprocessing representation also matches the saved original exactly.
