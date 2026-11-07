# Methodology, feature ablations and interpretation

## Prediction scope and source preservation

Predict an advertisement's asking price in KZT, not a verified transaction price or intrinsic valuation. The original 3,999-row CSV, eligibility rules, 2,553 clean ad IDs, price labels, order and 1,787/383/383 split are unchanged. SHA-256 `b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468`; the original Git blob hash is checked on load. City, dates and verified condition were not collected or fabricated. `legacy_data.py` preserves the previous feature extraction/preprocessing so the four original baselines and previous ANN are actual controls.

Source metadata and the original repository are documented in README. Kaggle's recorded update date and MIT catalog label do not establish individual observation dates or rights beyond the source metadata. No synthetic project rows were introduced.

## Deterministic semantic corrections

Cleaning still removes 527 new rows, 51 on-order rows and 868 duplicates after those filters. Features are deterministic per-row operations on the existing title/specification/description and original vehicle fields. They do not estimate parameters from validation/test labels, look up model specifications or use ad-specific exceptions.

The structured description prefix before `КПП` supplies recognized fuel in preference to the original parser's field. Recover 16 `electric` and 109 `gas_petrol` listings. Normalized gas–petrol mentions total 114, including 109 structured mentions and 5 narrative/structured disagreements. The latter retain structured petrol and expose `dual_fuel_claim`/`fuel_text_conflict`. Unknown fuel stays explicit. The audit key `narrative_dual_fuel_claims` counts all normalized positive mentions, not only the disagreements.

Engine size is recovered from the structured prefix with the retained 0.5–8 L bounds. For 15 explicit electric listings without positive literal displacement, 0 L plus `engine_not_applicable=1` denotes structural inapplicability. These are not unknown combustion sizes to replace with a median. The one electric/positive-engine source disagreement keeps its reported valid size with `engine_source_conflict=1`; no car-name guess adjudicates the conflict. The 2 genuinely missing/implausible combustion/hybrid sizes stay NaN with `engine_missing=1`. Their fitted medians come from training only.

## Enrichment is text evidence, not verified equipment

Full representation: 15 numeric and 7 categorical raw predictors. `brand_model` combines existing labels; trim uses explicit bounded AMG/GTS/GTI/M Sport/F Sport/GR Sport tokens, rejecting service-shop mentions and local negation. Positive restyling/leather/panorama/warranty flags require explicit matched wording. Source coverage: restyling_claim: 8, leather_claim: 205, panorama_claim: 53, warranty_claim: 6. Zero means no supported positive statement, not verified absence. Text is truncated; regex coverage and seller claims can be wrong.

Explicit power requires a number and a power unit. Russian лс/л.с. converts metric horsepower by 0.73549875; hp/bhp by 0.745699872; kW stays kW. Conflicting explicit values remain missing. Only 3 source rows have eligible power. Bare numbers, asking price, price_raw, URL and ad ID cannot generate power or other predictors. Test invariance checks change prices/identity while holding the description/specification fixed. Full descriptions are not fed as a text embedding or NLP model.

Corrected-only representation: year, log mileage, mileage-missing flag, cleaned engine and three engine-state flags (7 numeric), plus brand/model/body/corrected fuel/transmission (5 categorical). It drops extra claim flags, power/missingness, trim and brand_model. This is a useful controlled ablation of the extra feature family, but not a perfectly isolated causal estimate: feature-frequency and architecture/checkpoint selection also vary between reported rows. Do not attribute an observed score difference to one text token or fuel fix alone.

The preceding steering-pool representation adds one literal categorical `steering` field: 7 numeric + 6 categorical raw inputs, 307 encoded inputs with training-fitted OHE minimum frequency 2. `structured_features.py` requires a standalone `КПП` boundary and full-matches individual comma-delimited `Правый руль` or `Левый руль` clauses preceding it. It does not infer steering from brand/model, prices, IDs or URLs. Missing, narrative-only and negated wording stays `not_stated`; opposing exact clauses yield `conflict`. No matching clause is not evidence of left-hand steering. The independent TRAIN-only audit found 93 `right_explicit` and 1,694 `not_stated` among 1,787 rows, with no observed left-explicit/conflict category. It inspected no validation/test records and fitted no model; fixtures are behavioral checks, not new dataset observations.

## Fitting, tuning and selection

### Lossless source enrichment and quality metadata

The derived export has 2,553 × 47 columns and preserves every cell of the loader's 32 existing columns, row order, IDs, prices and original split. The unchanged raw 14-column CSV remains the source of truth. `dataset_features.py` adds seven candidate inputs, one age audit proxy and four row-local quality flags; strict steering and source-row/split metadata are also included in the export. `dataset/data_dictionary.csv`, `quality_report.json` and the 598-row `review_queue.csv` separate predictor candidates, targets, raw text, identities and review metadata. Review flags never repair/delete rows or replace values with guessed truth. `python dataset_export.py` reproduces the export; main training invokes it as well.

TRAIN source evidence finds 1,545/1,787 descriptions ending in `...` at lengths 120–121, strong evidence of capped snippets without proof of the upstream mechanism. `description_truncation_signal` applies a narrow trailing-ellipsis/119–122-character review rule; its absence does not prove complete text. Older low mileage, repeated-digit mileage and any present original drive label are review flags only. The original drive extraction can confuse equipment wording with drivetrain. Missing/suspicious mileages are retained, with no automatic ×1,000 correction or imputed true odometer. Similar listings do not establish vehicle identity without unavailable VIN/date/image evidence, so the existing ad-ID deduplication/split stays fixed.

The strict source parser requires exactly one complete comma-delimited КПП/transmission clause. It skips an optional full mileage clause, accepts exterior color only in the ensuing source slot, then reads a contiguous closed block of exact equipment labels; unknown, negated, truncated or narrative clauses stop it. TRAIN coverage: 1,281 literal colors, 642 alloy-wheel claims, 194 sunroof claims. `alloy_wheels_claim`/`sunroof_claim` are 1 only for positive exact clauses and NaN otherwise; no mention is unknown, not absence. Modeling candidate equipment requires train-fitted missingness indicators. Panoramic roof is distinct from literal sunroof. Upholstery `multiple_literals` is ambiguity for one category, not verified physical conflict. Color names remain literal, including source labels such as `вишня`, without invented paint codes/hues.

`model_family_lexical` uses narrow brand-scoped syntax: complete BMW three-digit names get numeric-prefix categories; Mercedes-Benz/Lexus names beginning letters plus a number get literal alphabetic prefixes. Unmatched names preserve their original brand/model identity. TRAIN has 254 matched rows and 384 lexical categories; this is not catalog equivalence, generation inference or verified technical identity. Raw model labels remain available.

`vehicle_age_proxy = 2025 − model_year`, with invalid/future/nonintegral years unknown; `log_vehicle_age` applies log1p. `log_annual_mileage_proxy = log1p(reported_mileage / max(age, 1))`; unknown/invalid mileage stays missing. These deterministic transformations use the fixed 2025 source snapshot, not the current year, first-registration dates or certified annual usage. Only these two new fields enter the selected follow-up; other candidate columns remain auditable research fields, and all four quality flags remain excluded from predictive inputs.

### Descriptive TRAIN utility study and separate validation follow-up

The utility protocol fixed five groups: base corrected-plus-steering, plus age/usage, plus lexical family, plus literal claims, and plus all seven. Three shuffled outer KFold folds within the original 1,787 TRAIN rows each use a 20% inner checkpoint split. A single seed-42 128→64 ANN, cap 150, patience 25, produces 15 fits. Inner training alone fits imputers, added numeric missingness indicators, scalers, OHE and standardized-log target; inner labels select the KZT-MAE checkpoint, then outer held-out predictions are made once. This differs from the main five-seed/fixed validation-test comparison and must not be numerically presented as the same benchmark.

Aggregate held-out TRAIN MAE is 2,074,609.5707 KZT for base and 2,029,914.5689 for age/usage; within ±20% is 57.1349% and 59.4852%. The pair improves both in all three folds. Lexical family, literal claims and all-seven groups instead have MAE 2,120,488.4086, 2,215,269.8859 and 2,237,870.7420 KZT. All negative results remain. These are descriptive outcomes under one seed/three folds, not a causal estimate of age/usage value or independent new-data evidence.

The original utility study explicitly excluded final-model promotion in its script. A later separately scoped pass, informed by that completed TRAIN evidence, tested **only the age/usage pair** on the existing validation set. It retains the same five seeds/128→64/300-epoch cap/45-epoch patience protocol, 309 encoded inputs, both MAE/dual checkpoints and fixed convex weights 0.25/0.50/0.75. Its 15 candidates comprise the prior robust baseline, two pools and 12 blends with either the baseline or unchanged forest. Strict dual eligibility and normalized MAE/miss ranking use the prior Robust ensemble as reference. This adds adaptive feature-family selection and further validation reuse; it is not a precommitted winner from the descriptive study. No other field group is tried on validation, and no new search follows the frozen selected historical-test evaluation.

Learned imputers, scalers, category vocabularies and minimum-frequency groups fit only on training observations or the current CV training fold. Enhanced categorical missingness uses constant `unknown`; legacy controls retain mode imputation and OHE minimum frequency 5. ANN target scaling sees training targets only. All final comparison weight fits retain the 1,787-row training set; CV and the utility study fit only their assigned training subsets. No train+validation refit changes the comparison protocol.

The enhanced KNN, tree and forest searches share shuffled KFold(n_splits=3, random_state=42), optimize negative MAE in KZT, and compare OHE minimum frequencies 1/2/5 within CV. KNN varies k=[3,5,9,15,25,35], uniform/distance and Manhattan/Euclidean; tree depth=[6,10,14,unlimited], leaf=[3,5,10]; forest uses 200 trees, max_features=[0.7,1.0], leaf=[1,3]. Expanding frequency multiplies the former candidate counts by three.

| Model | Candidate count | Training CV MAE ± SD, m KZT | Selected parameters |
|---|---:|---:|---|
| KNN tuned | 72 | 2.623 ± 0.104 | `{"model__n_neighbors": 5, "model__p": 1, "model__weights": "distance", "prep__cat__encode__min_frequency": 1}` |
| Tree tuned | 36 | 2.842 ± 0.141 | `{"model__max_depth": 10, "model__min_samples_leaf": 3, "prep__cat__encode__min_frequency": 1}` |
| Random forest tuned | 12 | 2.172 ± 0.142 | `{"model__max_features": 0.7, "model__min_samples_leaf": 1, "prep__cat__encode__min_frequency": 1}` |

Historical controls keep the original configurations; the original tree had unshuffled midterm CV, while new searches use shared shuffled folds. CV standard deviations describe only these three folds and are not confidence intervals.

ANN full-feature minimum frequencies are 1/2/5; corrected-only frequencies are 2/5. Each compares fixed hidden widths 64/32 and 128/64 under identical training settings. `ANN previous` is retrained on the unchanged legacy representation. Validation MAE chooses checkpoints, representation/frequency and the best ANN family; `ann_full`, `corrected_ann`, and `previous_ann` retain each report. The top-level `ann` describes the validation-best ANN row `ANN corrected`, which may differ from the literal full-feature table row `ANN`.

CatBoost compares six fixed candidates: depth 4/6/8 with raw-price MAE loss or log1p-price RMSE loss. Native categorical statistics are learned inside each actual fit; `one_hot_max_size=0`, numeric NaN retained, categorical missingness converted to `__missing__`. Each fold uses its own validation for patience-60 stopping, cap 1,000 iterations, lr 0.05, L2 leaf penalty 5, symmetric trees, seed 42. CV ranks original-target KZT MAE. The chosen configuration is fitted on training only with external validation controlling its final KZT-MAE checkpoint. Current choice: `catboost_log_rmse_depth8`, best iteration 993/1000. CatBoost internally rounded target buffers can make custom inverse-log history metrics differ by a few KZT; final/candidate scores are recomputed against original labels.

In the first improvement pass, six fixed convex blends compared the validation-best single ANN with CatBoost or Random Forest; ANN weights 0.25/0.50/0.75, partner weight 1−w. No regression was fitted to test targets. That pass selected ANN corrected × 0.75 + Random forest tuned × 0.25, validation MAE 2.289m KZT, under the **original MAE-only rule**. This delivered `Ensemble` and all 12 original comparison rows remain unchanged controls. The new dual-goal rule below is a later user-directed extension, not a retrospective redefinition of the original objective.

## Preceding steering-pool search and dual-metric rule

In the preceding accuracy round, the user requested both lower original-KZT MAE and higher within-±20% frequency. Development recorded 20 CatBoost configurations × 3 training folds and six validation finalists; 12 alternative configurations × 3 folds and five finalists; eight ANN pools; and a predeclared 30-blend grid. Including the two delivered controls, that validation ledger contains 51 candidates. This count excludes individual CV fits, per-epoch checkpoint searches and earlier rounds. All trials and failed finalists remain visible. No standalone CatBoost/alternative finalist improved both validation quantities over the delivered ensemble.

For validation hit fraction `h`, normalize against delivered-ensemble values `MAE_b` and `h_b` and define `D = 0.5 × [MAE/MAE_b + (1−h)/(1−h_b)]`. Eligibility requires **strictly smaller MAE and strictly larger h**. Rank eligible candidates by D; retain the baseline if none qualify. The development registry's deterministic tie breaks follow D, MAE and name. There was no deciding tie for the frozen winner. This rule values normalized error and misses equally; it is a declared preference, not an automatically optimal economic utility.

That round's frozen winner was 0.75 × corrected-plus-literal-steering dual-checkpoint ANN pool + 0.25 × the unchanged fitted Random forest tuned, averaged in original KZT. Validation MAE moved 2,289,371.6968 → 2,248,401.7334 KZT; within ±20% moved 246/383 → 254/383 (64.2298% → 66.3185%). The integrated entry point repeats this exact family: two pools, six fixed ANN/forest mixtures and the delivered control, nine selection candidates. These produced the preceding 15-row comparison and remain unchanged controls for the age/usage follow-up. It does not rerun the wider 51-candidate development search or its 32 new training-CV configurations, and does not tune new weights after test evaluation. Unselected research binaries are omitted; protocols, scores and validation predictions remain auditable.

Each pool contains all five seeds 11, 23, 42, 71 and 101, each weight 0.2; no subset selection. One fixed 128 → 64 ReLU architecture and one linear output use Adam lr 0.001, batch 64, L2 alpha 0.001 and training-standardized log1p KZT. Training cap is 300 epochs; stop 45 epochs after neither the MAE nor dual-score best checkpoint improves. Save both checkpoint networks per seed. MAE-checkpoint epochs are 56/54/72/24/21; dual-score epochs are 30/61/69/63/24; executed runs are 101/106/117/108/69 in seed order. Train input/target scalers each see 1,787 rows. `results/robust_ann/robust_ann_history.*`, per-seed checkpoints and pool reports expose convergence, validation losses and post-checkpoint changes. Optimizer loss is half squared error plus L2 in standardized-log space; the displayed MSE omits the half factor/L2 and is not KZT MAE.

The pool may reduce initialization variance, but a five-seed average does not prove robustness on new data. Both checkpoint searches and blend selection reuse validation labels. Training/validation gaps and post-checkpoint worsening are descriptive overfitting diagnostics on that reused set. `results/ann_history.*` and top-level report `ann` still describe the historical single-seed `ANN corrected` control; the new pool has its own explicit `robust_ann` report and histories.

## Experiment chronology and test-informed limitation

Historical test errors motivated the electric/dual-fuel/engine/text hypotheses. The first enhanced search selected full ANN at validation MAE 2.446m but test MAE 1.849m, versus previous ANN 1.755m. This worsened test performance despite its small validation gain. After seeing that outcome, the earlier follow-up added the corrected-only ablation and a fixed small blend grid. The full ANN row remains visible; that earlier pass's selection/weights used validation MAE only. The latest follow-up instead uses the strict-both eligibility and dual score described above.

This sequence means the historical test influenced research directions. Calling the final test “untouched,” “independent,” or a proof that the improvements generalize would be misleading. Freezing each batch before its test prediction prevents choosing that batch by its test ranking, but does not undo prior test inspection or adaptive hypothesis development. The single repeatedly used validation set also supports many choices; performance uncertainty is larger than one winning number suggests. New dated external validation is required for independent evidence.

In the latest round, the selected specification and component artifacts were frozen before one evaluation of that selected candidate on the known historical test, and the search stopped afterwards. No test labels were used for estimator fitting, checkpointing or candidate selection in this round. The unchanged common loader still reads the full source CSV for original eligibility checks before splitting; therefore “test labels were never accessed” would overstate the protocol. The declared retain-baseline decision if the frozen candidate failed either test goal also uses the known test for deployment acceptance. The full project rerun checks reproducibility of the frozen choice; it is not another search opportunity.

## K-Means and PCA subset

Displacement geometry uses 1775 non-electric training rows; 12 explicit electric training listings, including the source-conflicting record, are excluded only from this exploratory analysis. They remain in prediction and test evaluation. This is an intentional change from the previous all-train unsupervised analysis, not an identical before/after cluster comparison.

Fit train-median imputation and StandardScaler on year, log1p(mileage), engine. Log mileage reduces skew but defines the distance geometry. Price and categorical variables do not enter K-Means/PCA. Compare k=2..8, 20 initializations, seed 42; the highest silhouette selects k=4 (0.3686). This is a geometric heuristic, not evidence of universal market segments. Profiles are observed medians with nonmissing counts, not centroids or imputed means; post-hoc asking prices never drive fitting.

| Cluster | n | Median year | Observed mileage, km | Engine, L | Post-hoc asking price, m KZT |
|---|---:|---:|---:|---:|---:|
| 0 | 813 | 2015 | 146,000 | 2.0 | 7.690 |
| 1 | 245 | 2011 | 183,500 | 4.0 | 12.200 |
| 2 | 210 | 2023 | 23,900 | 2.0 | 14.125 |
| 3 | 507 | 1999 | 280,000 | 2.4 | 3.000 |

PCA variance shares: 47.67%, 32.46%, 19.87%. The first two retain 80.13% and the 2D view omits 19.87%. PC1 coefficients: year +0.691, log mileage -0.680, engine -0.246. PC2: year +0.129, log mileage -0.219, engine +0.967. These are signed unit component directions, not feature correlations; signs can flip. PCA supports visualization and is not a supervised price predictor. Missingness, omitted categories/scaling and a single seed limit cluster interpretation; no stability study is claimed.

## Historical single-ANN procedure, convergence and overfitting

Best historical single-seed ANN by validation: `ANN corrected`, feature set `corrected`, OHE minimum frequency 2, 305 encoded inputs, 128 → 64 ReLU hidden layers, one linear output. Train-only StandardScaler transforms log1p(price_KZT). Adam lr 0.001, batch 64, L2 alpha 0.001, beta1 0.9, beta2 0.999, epsilon 1e−8, seed 42; one partial_fit epoch per step, max 400, external-validation patience 40. Save/deep-copy the lowest KZT validation-MAE checkpoint and restore it.

Selected epoch 71 of 111 executed. Final validation MAE is 2.5% above the best; overfitting heuristic flag: `False`. Training transformed-MSE at first/best/final is 0.2259/0.0222/0.0191. The heuristic is final validation MAE >1.05×best and final train MSE below checkpoint train MSE. Its absence would not establish external generalization. Learning plots report ordinary transformed-space MSE, while the optimizer includes a half factor and L2. These units cannot be compared numerically with KZT MAE.

Log targets stabilize scale and can balance relative error, but do not optimize MAE in original KZT or guarantee improvement on expensive cars. Inverse predictions are guarded finite/nonnegative; no unbiased-mean back transformation or calibrated prediction interval is claimed.

## Latest frozen selection, metrics and uncertainty

The age/usage validation winner is **Usage ensemble = 0.25 × new MAE-checkpoint pool + 0.75 × prior Robust ensemble**. Validation MAE is 2,238,371.6080 KZT and within ±20% 259/383 (67.6240%), versus 2,248,401.7334 and 254/383 (66.3185%). The two new pool variants and 12 mixtures are validation tuning records; only the selected mixture is appended to the 15 previous test-comparison rows. The main entry point reproduces the frozen family and all old controls, not the broader TRAIN utility/CatBoost/alternative research searches.

The new pool has 309 encoded inputs: prior corrected+steering plus `log_vehicle_age` and `log_annual_mileage_proxy`. Five seeds, architecture/optimizer and 300/45 stopping settings match the retained 307-input pool. In seed order 11/23/42/71/101, new MAE checkpoints are 34/56/93/56/31, dual checkpoints 29/58/74/58/20, runs 79/103/138/103/76. The selected graph therefore has five new MAE networks at 0.05 each, five prior dual networks at 0.1125 each and the unchanged forest at 0.1875. No literal color/equipment/family or quality flag enters the selected graph. Pool histories and reports are saved separately under `results/usage_ann/` and `results/robust_ann/`; the old top-level `ann` history remains a historical single-ANN control.

The frozen historical-test comparison versus prior Robust has MAE 1,646,174.8055 → 1,634,563.5634 KZT (−11,611.2421 KZT; −0.70535%) and within ±20% 221/383 → 224/383 (+0.78329 percentage points). RMSE improves 3,246,982.8582 → 3,189,732.2921 KZT and R² 0.913477 → 0.916501; median APE improves 16.6283% → 16.6088%. Median absolute error instead increases 795,924.5401 → 801,611.0973 KZT, with one fewer hit each at ±10% (132→131) and ±30% (287→286). The retained median-baseline success criterion is met at 72.53% lower MAE. These are observed historical differences, not a causal feature effect or proof of broad improvement.

Paired bootstrap: NumPy `default_rng`, seed 20261107, 10,000 resamples of 383 listing indices with replacement, the same indices for both frozen predictions. Percentiles 2.5/97.5 give [−15,173; 41,640] KZT for `old MAE − new MAE` and [−0.7833; 2.6110] percentage points for `new hit rate − old hit rate`. Both include zero. They omit training variability, adaptive selection and possible listing dependence; they are descriptive conditional intervals, not unbiased post-selection guarantees, independent generalization evidence or calibrated per-car intervals. The preserved frozen evaluation under `results/dataset_enrichment_study/usage_validation/` is not recomputed by main training. The older steering-round bootstrap remains separately recorded under `results/accuracy_followup/`.

One frozen selected-winner historical-test evaluation preceded integration; both requested observed metrics improved and the candidate was retained under the disclosed test acceptance rule. This acceptance uses known test data. The main rerun reproduces that same specification, with no further search. Earlier test-informed hypotheses, the descriptive-study-informed age/usage choice, multiple checkpoint searches and repeatedly reused validation all remain part of the adaptive history. Calling the result independent or untouched would be inaccurate. No test labels are used for fitting/checkpointing/candidate ranking in this round, while the common loader still parses the source for original eligibility.

## Reading errors and practical limits

Median APE = median(abs(y−prediction)/y), not mean percentage error. Within ±10/20/30% are empirical fractions, not confidence levels. Positive residual means actual−prediction>0, or underestimation. Train quartiles at 3.6/6.7/11.5m KZT define evaluation segments, not predictors. Selected-model band MAE: Q1 0.783m (n=112), Q2 1.030m (n=100), Q3 1.057m (n=87), Q4 4.088m (n=84). Largest test error: 2022 Mercedes-Benz EQS SUV, actual 58.000m, prediction 30.208m, error 27.792m KZT. All error groups retain counts; tiny groups cannot support broad claims.

## Separate simplicity decision

The later [fixed simplicity protocol](experiments/simplicity_review/PROTOCOL.md)
uses only the original 1,787 TRAIN rows, with five outer assessment folds and
an inner checkpoint split in each fold. The three simpler recipes are compared
with a fold-local rebuild of the historical recipe, never the full-TRAIN saved
artifact. Preprocessing and models fit the same inner training rows. A/B share
one trajectory; their checkpoint criteria differ. Source/protocol/code hashes
and row identities were recorded before fitting; no new recipe or relaxed gate
was added after viewing results.

No simpler recipe satisfied the user's joint preference. The two-component
ANN/forest mixture was within the 3% global-MAE allowance (+0.6655%), but lost
23 overall ±20% hits and 6 affordable-car hits relative to the rebuilt reference.
Its Q1 hit count increased by one while Q2 lost seven. The fixed decision
therefore preserves the current model and produces no replacement artifact.
All four outcomes and segment counts are in `results/simplicity_review/`.

These OOF results assess different rows/training sizes from the historical
383-row test table; their absolute scores must not be read as a change in that
test's accuracy. Historical research feedback and OOF-based recipe selection
still prevent independent generalization claims. The original validation/test
were excluded from candidate fitting, checkpointing and prediction in this round.

Test maximum is 93.5m versus validation maximum 222m, so tail composition differs. Advertised 2025 prices, capped/unverified seller text, missing city/dates/detailed condition and sparse premium models limit use. The new export exposes those limits; it does not recover hidden ground truth or validate seller claims. Do not remove valid expensive ads, multiply suspicious mileage automatically, call the result current 2026 pricing or infer catalogue identity/causality. Estimates need human inspection for a real purchase. No geographical fairness guarantee, transfer guarantee or automated financial suitability decision is provided. Independent later-data validation and uncertainty calibration remain future work; no new rows are promised in this submission.

## Reproducing saved inference

The model bundle records its feature representation. An individual legacy model consumes legacy raw columns; enhanced individual models consume enhanced raw columns; a saved ANN also requires its saved preprocessor. A selected mixed bundle consumes aligned `{'legacy': frame_legacy, 'enhanced': frame_enhanced}` inputs and applies each component's own transformer. Apply both `structured_features.add_structured_features` and `dataset_features.add_dataset_features` to the original enhanced frame returned by `data.load_data` **before** selecting saved feature columns. The Usage ensemble requires `steering`, `log_vehicle_age` and `log_annual_mileage_proxy`. The dataset helper rejects generated-column collisions, so do not apply it twice to an already enriched export. Use the saved `features` dictionary to build both views in identical row order. Merely giving both views the same column names cannot reproduce their different fuel/engine semantics. Keep `data.py`, `legacy_data.py`, `ann.py`, `catboost_model.py`, `ensemble.py`, `robust_ann.py`, `structured_features.py` and `dataset_features.py` importable with the joblib bundle. README gives a complete historical-test inference recipe; no transformer is refitted at inference.

## Explicit EDA and regulation scope

The current entry point generates four training-only EDA plots and one interpretation per plot in `report.json` → `eda.plots`: prices, model-year medians, mileage/price and fuel-price distributions. The fuel boxplot retains all training listings, uses a labelled log-price scale and reports group counts; electric (12) and hybrid (22) samples are small and confounded by vehicle composition. See README for each interpretation and REGULATION_COMPLIANCE.md for the mapping to both course PDFs. The final submission uses the endterm 10–15-slide/12-minute format while retaining the earlier core work.
