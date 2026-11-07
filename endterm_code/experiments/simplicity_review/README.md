# Reproduce the TRAIN-only simplicity review

No tested simplification met the joint preference. The completed run retains
the historical recipe as its reference and creates no development candidate.
The original `results/selected_model.joblib` and historical results are
unchanged. A separate verifier checked the saved predictions, metrics and
acceptance gates; that check is not independent evaluation of model quality.

## Fixed comparison

| Recipe | Prediction rule | Inference components |
|---|---|---:|
| A: `single_mae` | One age/usage ANN; inner KZT-MAE checkpoint | 1 |
| B: `single_log` | The same ANN trajectory; inner absolute-log-error checkpoint | 1 |
| C: `single_forest` | 0.75 × A + 0.25 × fixed Random Forest | 2 |
| D: `historical_recipe` | Historical selected recipe rebuilt inside each fold | 11 |

Components are distinct estimators in the mixture, not training fits or the
trees inside the forest. D is a fold-local reconstruction, not a new score for
the saved historical artifact or a replay of its past search. The architecture,
seeds, folds and acceptance rule were frozen in `PROTOCOL.md` before fitting.

Only the original 1,787 TRAIN rows enter the review. Five outer folds give each
row one held-out prediction. A separate 20% inner checkpoint partition controls
epoch selection. Preprocessors and target scaling fit only on the remaining
inner training rows. The historical 383 validation and 383 test rows are
excluded. Source/protocol hashes and fold identities were saved before fitting.

## Completed development results

Values come from `../../results/simplicity_review/summary.json` and
`selection.json`; MAE is rounded to KZT. Sizes are median serialized fold-model
sizes in bytes, not sizes of a refitted candidate.

| Recipe | MAE, KZT | MAE change vs D | Global hits ±20% | Affordable median APE | Median bytes |
|---|---:|---:|---:|---:|---:|
| A | 1,959,116 | +4.297% | 1070/1787 (59.88%) | 20.00% | 1,273,257 |
| B | 1,987,068 | +5.785% | 1074/1787 (60.10%) | 20.06% | 1,273,387 |
| C | 1,890,899 | +0.665% | 1101/1787 (61.61%) | 19.06% | 21,296,802 |
| D | 1,878,399 | reference | 1124/1787 (62.90%) | 18.77% | 32,608,659 |

All cells below are within-±20% hit counts. Bands use actual KZT prices:
Q1 ≤3.6m; Q2 (3.6m,6.7m]; Q3 (6.7m,11.5m]; Q4 >11.5m.

| Recipe | Q1 /452 | Q2 /453 | Q3 /440 | Q4 /442 | Affordable ≤6.7m /905 |
|---|---:|---:|---:|---:|---:|
| A | 191 | 262 | 308 | 309 | 453 |
| B | 186 | 265 | 317 | 306 | 451 |
| C | 204 | 261 | 320 | 316 | 465 |
| D | 203 | 268 | 326 | 327 | 471 |

## Decision and limits

Relative to D's pooled OOF predictions, A/B/C must pass every gate:

1. Global MAE ≤1.03 × D's MAE.
2. Global within-±20% count ≥D's count.
3. Q1 and Q2 hit counts each ≥D, with a strict gain in at least one band.
4. Median absolute percentage error at ≤6.7m KZT ≤D.

Eligible candidates are ordered by component count, then affordable median
APE, global MAE and name. None qualified. A and B exceed the 3% MAE limit.
C stays within it, but loses 23 global hits and seven Q2 hits, and worsens
affordable median APE. Its one extra Q1 hit is insufficient. The gates are
unchanged; no further search follows this result. True-price bands are used
only for assessment and stratification, never as inputs or inference gates.

These are development OOF results: all source data already influenced past
project decisions, and this comparison selects among recipes. They are not
independent new-data estimates and must not be compared directly with the old
383-row test metrics. A one-hit difference does not establish a statistically
reliable gain. Fold metrics and histories, `oof_predictions.csv`, provenance,
fold identities and protected-artifact hashes are retained in the run folder.

No `development_candidate.joblib` or `candidate_metadata.json` was created.
Under the protocol, a qualifying recipe would be refitted on all TRAIN using
the rounded median selected fold epochs. Its saved object would be a direct
`PredictionEnsemble`, with its contract in `candidate_metadata.json`: supply
aligned DataFrames in `{'enhanced': enhanced}` and, for C, also `'full': full`.
Here `full` is the vehicle-engineered frame; `enhanced` adds
`add_dataset_features(..., snapshot_year=2025)` and `add_structured_features`.
Use the saved transformers without fitting at inference. Such a development
refit still needs independent evaluation before any historical replacement.

## Reproduction and course scope

From `endterm_code`, in the project's Python 3.12 environment, verify the saved
evidence without training:

```bash
python experiments/simplicity_review/verify_simplicity_review.py --project-source . --output-dir results/simplicity_review
```

For another run, use a fresh directory in both commands:

```bash
python experiments/simplicity_review/run_simplicity_review.py --project-source . --output-dir results/simplicity_review_new
python experiments/simplicity_review/verify_simplicity_review.py --project-source . --output-dir results/simplicity_review_new
```

Existing output directories are refused. This optional review is separate from
`python run_project.py`. The required median/Linear/KNN/Tree baselines, CV
tuning, ordinary ANN, EDA, K-Means and PCA remain part of the course project.
The source is a 2025 asking-price snapshot, not 2026 sales data. See
`../../SIMPLICITY_REVIEW_RU.md` for the Russian explanation.
