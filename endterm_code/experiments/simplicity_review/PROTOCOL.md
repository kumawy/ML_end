# Fixed TRAIN-only simplicity comparison

Frozen before candidate fitting on 2026-11-07. This is development evidence,
not a new independent test or an unbiased evaluation of the selection procedure.
All data have already influenced historical project decisions. The historical
383-row validation and 383-row test are excluded from this experiment.

## Data boundaries

Use exactly the original 1,787 TRAIN rows from `data.split_data`, preserving the
source CSV, labels, rows and split. Order these TRAIN rows by original source
index before constructing the deterministic folds. Fixed price bands have upper bounds 3.6m,
6.7m and 11.5m KZT; these were previously obtained from TRAIN quartiles. They
are evaluation/stratification labels only, never predictors or inference gates.
Five outer StratifiedKFold folds: shuffle=True, random_state=20261107. Each
outer training partition has an inner stratified 20% checkpoint partition,
random_state=20261108. All models fit the same inner training rows. Every
preprocessor and target scaler is fitted only on these permitted fitting rows.
Outer targets are used solely for the once-per-row out-of-fold assessment.

## Four fixed recipes (no new search)

All ANN use MLPRegressor: (128,64), ReLU, Adam, alpha=.001, batch_size=64,
learning_rate_init=.001, shuffle=True, early_stopping=False, max_iter=1,
tol=0; target is standardized log1p(KZT), inverted for price predictions.
Input numeric imputation/scaling and categorical encoding follow existing code.

A. `single_mae`: corrected inputs + literal steering + fixed-2025 log age and
log annual mileage proxy; OHE min_frequency=2; network seed42. Lowest inner
KZT MAE checkpoint.

B. `single_log`: same single trajectory as A; lowest mean absolute log1p price
error checkpoint. It emphasizes relative error, without an arbitrary price gate.
A/B run at most300 epochs and stop45 epochs after neither checkpoint improves.
Strict improvement updates checkpoints; ties retain the first.

C. `single_forest`: fixed .75*A + .25*RF. RF uses existing full feature view,
OHE min_frequency=1, n_estimators=200, max_features=.7,
min_samples_leaf=1, random_state=42, n_jobs=2; raw KZT target.

D. `historical_recipe`: rebuild the selected historical recipe inside each
fold. This is neither the original saved artifact nor its complete past search.
First train a corrected (no steering) reference ANN, OHE frequency2, seed42,
max400 epochs, patience40, inner KZT MAE checkpoint. Reference=.75*ANN+.25*RF.
Then existing default robust_ann five-seed (11,23,42,71,101) corrected+steering
pool, frequency2, max300/patience45, both MAE and normalized dual checkpoints;
reference controls its dual score. Robust=.75*dual pool+.25*RF. Next existing
age/usage five-seed pool uses Robust as dual-score reference and retains its
MAE pool. D=.25*usage MAE pool+.75*Robust. Preserve both historical checkpoint
criteria in the stopping rule even when a checkpoint is not used at inference.
No blend or architecture selection is rerun. Flat weights: each usage ANN .05,
each corrected dual ANN .1125, RF .1875.

## Eligibility and decision

User permits up to3% relative increase in average KZT error for simplicity and
better affordable-vehicle performance. Compare pooled OOF predictions with D.
A/B/C is eligible only if all conditions hold:

1. Global MAE <=1.03*D MAE.
2. Global count within +/-20% >=D count.
3. Counts within +/-20% in Q1 (<=3.6m) and Q2 (3.6m,6.7m] each >=D,
   with strict improvement in at least one.
4. Median absolute percentage error for affordable vehicles (<=6.7m) <=D.

Rank eligible variants by number of distinct inference estimators (A/B=1,
C=2,D=11), then affordable median APE, global MAE, name. If none qualifies,
retain D and report that no tested simplification met the joint preference.
Do not relax gates, add candidates or retune after observing OOF results.
Report all candidates, sample/hit counts, fold metrics and artifact size; a
one-hit difference is not evidence of a statistically established improvement.

If a simpler candidate qualifies, refit only that candidate on all1,787 TRAIN
rows, using the rounded median of its selected inner epochs across five folds
(for C use A's epochs). This fixed heuristic involves no old validation/test.
Store a separate development candidate with source/protocol hashes and input
contract. Its refitted artifact has no independent evaluation. Preserve the
current selected_model.joblib, historical benchmark files and course models.
Record source hashes and fold identities before fitting; preserve completed
evidence and refuse overwriting an existing run directory.
