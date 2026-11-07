# TRAIN-only original-dataset quality audit

The strongest remaining enrichment is **explicit data-quality metadata**, not invented values or more rows. The three actionable gaps are uncertain mileage, truncated text, and uncertain vehicle identity across distinct advertisements. All original rows and values should be retained.

## Scope and reproducibility

- Original source: `kolesa_cars_2025.csv`, SHA-256 `b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468`.
- Existing `data.load_data()` retains 2,553 unique used-car advertisements from 3,999 source rows. Existing `data.split_data()` supplies the same **1,787 TRAIN rows**.
- This audit drops `price` and `price_raw` immediately after the existing loader/split. No target associations, fits, validation/test exploration, external data collection, source changes, split changes, or model changes were performed. The loader's existing basic price-validity inclusion check remains unchanged.
- The source snapshot is 2025. Age-review rules use 2025, not the current year. A model year is not a precise registration age.
- Run `run_audit.py` with the project environment. `evidence.json` contains counts, source-row indices and advertisement IDs for examples; `train_quality_flags.csv` contains only TRAIN metadata. No output contains prices.

## 1. Mileage is missing or ambiguous in a material minority of records

| Observation | TRAIN count | Share |
|---|---:|---:|
| Missing numeric mileage | 349 | 19.53% |
| Model year ≤2015 and advertised mileage ≤1,000 km | 11 | 0.62% |
| Integer mileage comprising one digit repeated at least three times | 43 | 2.41% |
| Missing numeric mileage but narrative contains a mileage word | 2 | 0.11% |
| Union of these review conditions | 397 | 22.22% |

These conditions overlap. Repeated digits can be genuine odometer readings, and an old car can be restored or rarely used. They are review signals, not proof of a placeholder, fraud, wrong units, or inaccurate mileage.

Examples copied from the original TRAIN descriptions:

- Ad `206855782`, Mercedes-Benz E 300, 1995: `с пробегом 555 км`.
- Ad `206840165`, Audi 100, 1991: `с пробегом 777 км`.
- Ad `206802268`, Mercedes-Benz E 200, 1992: `с пробегом 888 км`.
- Ad `206542242`, ВАЗ (Lada) 2113, 2011: `с пробегом 999 км`.
- Ad `181818823`, Toyota Camry, 2015: `с пробегом 257 км`.
- Ad `204181284`, Kia K7, 2021, missing numeric mileage: the seller says the car was ordered from Korea in 2023 `с пробегом 88 тыс.`. That is a historical claim; it does not establish the current odometer.
- Ad `198245912`, Mercedes-Benz E 200, 2016, missing numeric mileage: `Пробег родной 70 000`. The visible snippet lacks an explicit unit and remains a seller claim.

**Action:** add row-local review fields `review_old_low_mileage`, `review_mileage_repeated_digits` and `review_mileage_narrative_without_numeric` to the enriched artifact. Keep the existing `mileage_missing` metadata. Preserve missing values and all original numeric mileage values; do not multiply small values by 1,000 or replace repeated digits automatically.

An extraction consistency check found **1,438/1,438 available numeric mileages exactly match the first visible explicit `… км` value** in the snippet. There are no missing numeric mileages with such an explicit visible `… км` value. The suspicious small values originate in the visible source wording, rather than a demonstrated numeric-parsing bug. Recovering a current true odometer would require evidence unavailable in this dataset.

## 2. Descriptions are snippets, so unmentioned features are unknown

**1,545/1,787 descriptions (86.46%) end with literal `...`.** Every one is either 120 characters (186 rows) or 121 characters (1,359 rows); the maximum description length is 121 characters. This is strong internal evidence of a capped snippet field. The exact upstream truncation mechanism is not established.

Examples include ad `206840165` ending `без вложе...`, and ad `206542242` ending `ветровики, хру...`. Text after these endpoints is unavailable. This limits additional equipment, condition, registration, fuel-conversion, and power extraction. Only 3 TRAIN descriptions currently yield an explicit power value.

**Action:** add `review_description_ellipsis = desc.rstrip().endswith('...')` and `description_characters = len(desc)` as provenance/coverage metadata. Keep each extracted feature explicitly named as a *claim*. A zero text-claim indicator means “no supported positive statement visible,” not “the car does not have that feature.” Do not complete truncated words, infer unseen equipment from model/brand, or transform seller statements such as “без ДТП” into verified accident history.

## 3. Similar records do not establish the same vehicle

Within TRAIN, **62 records in 28 groups (3.47%)** share brand, model, year, mileage, cleaned engine size, fuel, body, and transmission. **52 of these 62** lack mileage, so most collisions are weak evidence arising from common specifications and missing information.

The strongest collision consists of two distinct ad IDs, `205015366` and `205015860`: both advertise a 1995 Nissan Terrano, 350,000 km, 2.7 L diesel, automatic, right-hand steering, and the exact same visible description ending `Машина в хорошем техническом...`. The available dataset contains no VIN, registration identifier, complete narrative, listing date, or image evidence to establish whether this is the same car, a repost, or separate cars using the same text.

**Action:** retain the records. Add TRAIN-local audit flags for repeated attribute combinations and repeated attributes plus description. These are audit-only fields and must not become model predictors or justify automatic deduplication. Do not alter the established split. Any future identity resolution would need an explicit evidence-based policy and a new predeclared evaluation protocol.

## Checks that do not justify further changes

- Raw TRAIN brand, model, fuel and transmission contain **zero** NFKC/whitespace changes and **zero** distinct-value collisions after whitespace normalization and case folding. There is no measured TRAIN benefit from a speculative alias merge.
- Model years range from 1956 to 2025; all satisfy the existing source-snapshot bounds. The 1956 car with 200 km remains a review case, not an invalid year or a deletion candidate.
- Raw engine size is missing in 505 TRAIN records, while the existing conservative text/electric handling leaves only 1 cleaned engine value missing. That remaining record is ad `204501352`, BYD Tang, whose visible source explicitly states `0.1 л, бензин`; preserve the report and the existing invalid-engine handling rather than infer the intended powertrain.
- Raw drive is missing in 1,764 rows and equals `awd` in 23. The known bare-word `полный` extraction is not trustworthy: visible examples say `полный электропакет` or `Полный комплектация`. Fuel/engine correction, explicit steering, and the separate drive parser review are already handled by the parent workflow; they are not new findings or reasons to restore this raw drive field as truth.

## Interpretation and next step

`candidate_flag_definitions.json` gives deterministic definitions and TRAIN counts. The flags support a reviewable enriched copy of the same real dataset. They do **not** certify seller claims, impute ground truth, add synthetic rows, remove cars, or establish improved prediction accuracy. No new model experiment is justified solely by this audit; any later experiment must predeclare feature treatment and use TRAIN-only evidence under the existing known-test limitation.

The original CSV SHA-256 was checked again after the audit and is unchanged.
