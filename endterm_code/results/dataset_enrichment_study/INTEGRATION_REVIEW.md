# Independent final integration review

**Pass for the completed code, dataset, inference bundle, saved-result reproduction and final documentation consistency. No unresolved finding remains from this review.** The integration results below are verified from completed current artifacts, not inferred from a progress log.

The review performed no training, feature proposal, candidate expansion, test-driven selection or additional model tuning. It verified the pre-existing frozen results and predictions. The earlier enrichment-only review remains a historical snapshot; its dictionary refresh issue is now resolved, and its unchanged-old-model statement predates this separately scoped integration.

## Scope and selection boundary

- Only `log_vehicle_age` and `log_annual_mileage_proxy`, the pair supported by the completed TRAIN-only study, are added to the usage ANN representation. Both are deterministic fixed-2025 transformations; no current year, external specification, new advertisement or synthetic row is introduced.
- `robust_ann.py` exposes a call-local, duplicate-rejecting allowlist for those two names. Target, identity, arbitrary columns and all new literal/quality fields are rejected as additional numeric inputs. Default module constants, the original two pools, and the original nine-candidate blend family remain intact.
- The usage pass explicitly enables the fifteen-candidate family: prior Robust ensemble, two usage checkpoint pools, and twelve fixed pool/reference blends. The references are the unchanged forest and prior Robust ensemble; weights are 0.25/0.50/0.75. Eligibility requires strict validation MAE reduction and within20 increase; ranking uses the existing equal normalized MAE/miss score. No test inputs are accepted by the fitting/selector APIs.
- In `run_project.py`, the selected model/configuration is written to `selection_before_test.json` before the test prediction block. The integrated validation winner is exactly the frozen winner: **0.25 × usage MAE pool + 0.75 × prior Robust ensemble**, validation MAE **2,238,371.607965701 KZT**, within20 **259/383 = 0.6762402088772846**.
- Every usage validation prediction and all fifteen candidate predictions, MAEs, hit rates and dual scores reproduce the frozen research artifacts **exactly**. The research pretest-selection hash also matches the frozen evaluation metadata.

## Controls and final saved inference

The comparison has **16 model rows**, with Usage ensemble selected. All **15 preceding controls** retain bit-exact validation and historical-test prediction columns compared with the original project results. Their IDs, row orders and target values also match.

The selected bundle has **25 enhanced feature names and 9 legacy names**. Its enhanced schema contains only the old 22-field contract, steering, and the two usage proxies. All four new quality flags, new lexical/color/upholstery fields, new equipment fields, raw age audit column, target text, IDs, URLs, raw descriptions and split/provenance metadata are excluded. Existing numeric missing-mileage semantics remain available; missing equipment candidates are not used by this final model.

The saved mixture is flat and has **11 components**:

| Component group | Members | Weight per member | Total weight |
|---|---:|---:|---:|
| Usage MAE-checkpoint ANN pool | 5 | 0.05 | 0.25 |
| Prior steering dual-checkpoint ANN pool | 5 | 0.1125 | 0.5625 |
| Unchanged fitted random forest | 1 | 0.1875 | 0.1875 |

Weights sum to one within floating arithmetic. The five usage components consume 9 numeric + 6 categorical source inputs; the five previous pool components retain 7 numeric + 6 categorical inputs. Saved preprocessors select their declared columns; the forest retains its original full representation. There is no nested ensemble ambiguity.

Freshly loading `selected_model.joblib`, deriving `add_structured_features()` and `add_dataset_features()` before column selection, selecting the bundle's declared enhanced/legacy columns, and predicting with the saved transforms reproduces every main historical-test prediction **exactly**. No transformer is refitted. Usage report paths resolve under explicit base directory `usage_ann`, and both declared pool artifacts exist there.

## Dataset/export preservation

Main calls the deterministic export before training. The completed export remains **2,553 × 47**, with every original retained 32-column cell and missing-value mask exact, original row order/identities/targets unchanged, and exact **1,787/383/383** split assignment. The unchanged original CSV hash is `b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468`.

The regenerated dictionary correctly identifies KZT targets and `raw_target_text`. Export status now states that exporting itself performs no fitting/selection and directs readers to project model results; it no longer incorrectly states that the final selected model cannot consume any new field. Candidate status does not imply every exported candidate is enabled. Source/literal claims and quality flags retain the earlier reviewed unknown/review semantics.

## Frozen result and uncertainty verification

The completed main predictions match the frozen `before_prediction`/`after_prediction` CSV **exactly**. Independently recomputing all saved metrics produces zero difference from `frozen_evaluation.json`.

| Frozen historical comparison | Before: prior Robust | After: Usage |
|---|---:|---:|
| MAE, KZT | 1,646,174.8054945795 | 1,634,563.563433875 |
| Within ±20% | 221/383 | 224/383 |
| Median absolute error, KZT | 795,924.5400783457 | 801,611.0973327365 |
| Within ±10% | 132/383 | 131/383 |
| Within ±30% | 287/383 | 286/383 |

The requested MAE and within20 measures improve slightly on this fixed historical split; the other listed measures worsen. The result is not a universal accuracy improvement.

The bootstrap is reproducible from the saved CSV using `numpy.random.default_rng(20261107)`, **10,000 paired draws** of the 383 listing indices with replacement, and plain 2.5th/97.5th percentiles of mean paired gains. Both recomputed intervals match the saved JSON **exactly**:

- MAE reduction: **[−15,173.169213397076; 41,639.709250695094] KZT**.
- Within20 gain: **[−0.7832898172323759; 2.610966057441253] percentage points**.

Both include zero. These are conditional comparisons of fixed fitted predictions on a reused historical split; they exclude training variability and adaptive-selection uncertainty. The source and reports disclose prior test-informed development. The verification does not establish independent generalization or justify restarting search.

## Independent verification artifacts

- `verify_integration_readonly.py`: reproducible checks using only existing source/exports, saved models and saved predictions; no fitting/tuning.
- `INTEGRATION_REVIEW_CHECKS.json`: detailed preservation, old-control parity, all fifteen candidate comparisons, bundle schema/component contracts, frozen metrics, bootstrap parity and source/code/artifact hashes.
- `INTEGRATION_REVIEW_TESTS.log`: **24 passing no-fit tests**—19 dataset/export contracts, four selector contracts and the additional-input rejection boundary. The parent's broader 58-test run is separate evidence and was not rerun as model training here.

## Final six-document consistency review

A final read-only pass covered `README.md`, `METHODOLOGY.md`, `EXPERIMENT_LOG.md`, `REGULATION_COMPLIANCE.md`, `CONTRIBUTIONS.md` and `DEFENSE_NOTES_RU.md`. Current sixteen-row/Usage selection, 58-test and 122.9-second claims are separated from historical stages. Seven export candidates are distinguished from the selected pair; fifteen TRAIN fits are distinguished from fifteen later validation candidates. The 307/309-input pools, ten ANN plus forest weights, failed feature groups, metric regressions and zero-crossing conditional bootstrap intervals are consistently explained. Student contributions remain explicitly unverified rather than invented.

One localized README ambiguity initially named the delivered ensemble as the generic dual-score reference for both rounds. Its owner corrected the text to each round's retained reference, explicitly mapping steering to `Ensemble` and usage to `Robust ensemble`; a fresh read verifies the correction. No remaining semantics/chronology finding was identified. The fifteen timed defense headings independently total **700 seconds**, leaving twenty seconds within the stated twelve-minute presentation limit.

No production code, dataset or six submission documents were edited by this reviewer. No post-test substitution or further feature hypothesis is recommended or performed.
