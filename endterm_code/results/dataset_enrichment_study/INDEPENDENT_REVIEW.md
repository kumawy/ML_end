# Independent enrichment review

**Disposition: implementation and dataset-integrity checks pass. The generated-dictionary refresh finding was subsequently resolved.** No parser, identity-preservation, split-preservation, or predictor-leakage blocker was found in the reviewed code. No source code was changed by this reviewer.

This report records the initial enrichment-only snapshot, before the later separately scoped age/usage model integration. Its statements about the old selected artifact being unchanged describe that earlier snapshot. See `INTEGRATION_REVIEW.md` for the final model/artifact review.

## One actionable finding

**[P2, resolved] The initial generated data dictionary predated the final exporter.** In the initially inspected `code/dataset/data_dictionary.csv`, `price` had generic `source units` and a fallback definition, and `price_raw` had role `raw_source_text`. The exporter already correctly declared KZT and explicit target semantics. The completed integrated main run regenerated the dictionary; an independent fresh read verifies `price` uses KZT and `price_raw` uses `raw_target_text`. The actual candidate-feature list excluded both fields throughout, so there was no observed target leakage.

## Evidence and checks

- Actual enriched CSV: **2,553 rows × 47 columns**, containing the original 32-column retained pipeline frame, one explicit steering field, twelve new feature/quality fields, and two split/provenance columns.
- Every original retained cell and missing-value mask matches the CSV round trip. Numeric values show zero observed round-trip differences; original price targets match exactly. Original ad IDs, row order, source-row indices, eligibility and split labels are conserved.
- Split assignments match the existing `split_data()` exactly: **1,787 TRAIN / 383 validation / 383 test**. No row or ad ID was synthesized or removed.
- The source CSV SHA-256 remains `b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468`. The staged source CSV, `data.py`, `structured_features.py`, `robust_ann.py`, `ensemble.py`, and selected-model artifact are byte-identical to the existing original project versions at review time.
- The 30 declared candidate predictors exclude target columns, target text, names/descriptions, IDs/URLs, raw drive, source condition, provenance/split metadata, the raw age audit field, and all four quality flags. The exported report and dictionary cover all 47 fields without duplicate dictionary entries.
- The review queue contains **598 distinct original advertisements**, all within the enriched dataset. Suspicious mileage and raw drive values remain unchanged; review reasons do not certify an error or vehicle condition. Description truncation is coverage metadata and is not, by itself, a queue reason.
- The cap signal matches the original TRAIN audit: **1,545 TRAIN rows**. The remaining text is not reconstructed.
- TRAIN lexical-family output contains **384 distinct labels**; **254 rows** receive one of the narrow prefix-group representations. These quantities measure different things. Unmatched model strings remain brand-scoped and distinct; no catalog equivalence or vehicle specification is inferred.

## Parser and inference contracts

`add_dataset_features()` requires only the original description, brand, model, year and mileage fields. It does not read a target, identifier, URL or raw price wording. It returns a copy, preserves source cells and row indexes, rejects generated-name collisions, fixes the default snapshot at 2025, and preserves missing/invalid proxy inputs as unknown.

Color extraction requires an exact known label at the exterior slot immediately after one complete transmission clause and an optional complete mileage clause. Equipment extraction follows a contiguous closed source-clause vocabulary and stops at an unknown, narrative, negated or truncated clause. Multiple source labels remain `multiple_literals`; the code does not guess a standardized color or mixed upholstery specification. Literal panoramic-roof wording does not invent a sunroof claim.

Equipment outputs contain only **1.0 or missing** in the actual CSV. Missing does not become a verified zero. The README and field definitions correctly require a training-fitted missingness indicator if numeric equipment fields are later modeled. The bounded utility-check source uses `add_indicator=True` and `keep_empty_features=True` for new numeric fields. This review did not fit or compare models, inspect held-out performance, or authorize another feature hypothesis. Existing production preprocessing does not acquire the new columns merely because they are present in the export.

The exporter rejects overlapping/incomplete splits, missing/duplicate ad identities, invalid retained targets and reserved metadata collisions. The actual export is lossless. It validates IDs and targets on CSV reload; the independent check additionally covered all original columns, indexes and split labels.

## Verification performed

Independently ran:

```text
python -m unittest discover -s tests -p 'test_dataset*.py' -v
Ran 19 tests ... OK
```

These cover source-cell preservation, target/identity independence, minimal inference input, proxy semantics, invalid/missing inputs, source boundaries, negation/truncation, multiple literals, lexical-family scope, row-local flags, name collisions, empty input, CSV identity/target/split round trips, and candidate exclusions. The parent's broader 54-test result was not rerun or counted as independent evidence here.

`INDEPENDENT_REVIEW_CHECKS.json` records the detailed column-preservation checks, source/code hashes, artifact checks and the generated-dictionary refresh finding. Review reads of held-out target values were limited to equality checks for preservation; there was no target association or performance analysis.

## Nonblocking readability note

The new fields have useful literal/unknown semantics. Some pre-existing pipeline fields still use the fallback “see data.py and METHODOLOGY.md” definition. A self-contained dictionary would be clearer with short definitions for fields such as corrected fuel, cleaned engine size and missingness indicators; this does not change any reviewed data or predictor behavior.
