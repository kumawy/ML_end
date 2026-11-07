# Independent review before this round's test evaluation

Verdict: **no blocking correctness or leakage issue found in the frozen validation selection.** `blend_44` is the winner under the declared rule. This approval concerns the validation selection and artifact reconstruction; it does not establish improved historical-test performance or new-data generalization.

## Independently verified results

Using the existing Python environment, recomputed every metric from `all_validation_predictions.csv` independently of the experiment's `stats` function. All 383 targets are positive and finite; IDs are unique and exactly match the delivered validation IDs and order. Predictions are finite and nonnegative. All 19 new individual columns match their family prediction CSVs, and every one of the 30 blend columns reconstructs from its recorded component weights within CSV floating-point precision.

| Validation quantity | Delivered ensemble | Frozen blend_44 |
|---|---:|---:|
| MAE, KZT | 2,289,371.696786993 | 2,248,401.7333698105 |
| Within ±20% | 246/383 = 64.22976501305483% | 254/383 = 66.31853785900783% |
| Median absolute percentage error | 0.1400908308954266 | 0.13885666330915522 |
| Normalized dual score | 1.0 | 0.9618550578394909 |

The observed validation change is **−40,969.9634 KZT MAE (−1.7896%) and +2.0888 percentage points within ±20% (net eight additional hits)**. These are descriptive validation differences after selection.

The registry contains 51 evaluated candidates: two delivered controls, eight ANN pool candidates, five alternative finalists, six CatBoost finalists, and 30 blends. Exactly eight strictly improve both validation metrics. Independently applying strict eligibility, then sorting by dual score, MAE and candidate name, selects `blend_44`. The per-family top-two shortlist also matches the declared score and name tie break. Its recorded weights mean 0.75 × corrected-plus-literal-steering ANN dual-score pool + 0.25 × delivered random forest, averaged in original KZT.

## Artifact, data and split checks

Reproduced the saved selected ANN, delivered RF, delivered ensemble and final blend predictions on **validation predictors only**, with agreement within 1e-6 KZT. The RF was extracted from the unchanged delivered ensemble; it is the original fitted random-forest pipeline. The ANN contains exactly the predeclared seeds 11, 23, 42, 71 and 101 with equal 0.2 weights. Each saved checkpoint agrees with that seed's minimum dual-score epoch in history: 30, 61, 69, 63 and 24 respectively. Target standardization matches training log1p prices and has 1,787 samples. Predictor scaling also has 1,787 samples, and fitted categorical columns include only the corrected view plus steering.

Recreated the seed-42 1,787/383/383 split from the original eligible source-row/ID manifest, without using labels. Source and scratch copies of `data.py`, `legacy_data.py`, `ann.py`, `ensemble.py`, `catboost_model.py`, `unsupervised.py`, `run_project.py`, both requirements files, original CSV and split manifest match the pre-round delivery SHA-256 manifest. Thus this experiment has not changed the original row eligibility, data, split or required control-method implementation.

For artifact prediction reconstruction, the raw CSV was read with `price` and `price_raw` excluded via `usecols`, then immediately restricted to validation source rows. Training-price checks used the existing train-only pickle. No test prediction/result file or raw test target was opened or inspected in this review. Machine-readable check evidence is in `REVIEW_CHECKS.json`.

## Extraction and protocol review

The steering extractor is deterministic and row-local. It requires a КПП boundary and exactly matches isolated comma-delimited `Правый руль` or `Левый руль` clauses preceding it. It reads neither prices nor IDs nor inferred brand/model specifications. Missing, negated, narrative-only or ambiguous wording is not silently assigned left-hand steering; conflict has its own category. The train-only audit's 93 right-explicit and 1,694 not-stated values reproduce. The selected ANN uses none of the broader narrative feature claims.

ANN predictors and target transformations are fitted on training only, while validation selects per-seed epochs and the final mixture. Alternative target encoding is cross-fitted inside training and CV folds. CatBoost early stopping is confined to training CV; final tree counts are fixed from those folds, and external validation does not fit CatBoost. The fixed blend grid was recorded before its results, with 30 blends and no target-dependent per-row gating or evaluation-row deletion. File timestamps are consistent with this ordering, although local timestamps alone are not an immutable preregistration record.

## Reporting limits and integration requirements

1. Validation is a development set: per-seed checkpoint searches, family shortlists, 51 final candidates, and prior experiments all reuse it. The net eight-hit improvement cannot be presented as independent confirmation, a significance result, or an unbiased estimate of future gain. Counting 51 final predictions also does not count every checkpoint or training-CV trial.
2. The historical test was known before this round. Even a paired bootstrap interval from its one frozen evaluation is conditional, descriptive uncertainty for this reused split; it does not correct adaptive historical selection or establish external generalization. Keeping the baseline if the frozen candidate fails the test's both-metrics rule adds a disclosed test-based deployment decision. Do not reopen the search after seeing this outcome.
3. Prefer the precise claim **“no test predictions or test-label use for fitting/checkpointing/candidate selection in this round”** over literal “test labels were never accessed”: the unchanged common loader parses the full source CSV and applies original eligibility before splitting. This is not additional leakage relative to the baseline, but literal non-access wording overstates what the scripts do.
4. Scratch training uses runtime monkeypatches and source-directory imports. Product integration still needs a stable module, deterministic five-seed checkpoint logic, a clear saved-inference contract, and behavioral tests that reject misalignment/invalid inputs. The reproduced validation artifact is evidence for the frozen choice, not a substitute for that integration review. Preserve all 12 reported controls and KMeans/PCA when integrating.
5. Freeze or hash the selected specification and component artifacts before evaluation/integration so later product changes cannot silently redefine the selected model. Keep this round's 51-candidate ledger and the historical-test disclosure in the final report and presentation.

No product files were edited by this review.
