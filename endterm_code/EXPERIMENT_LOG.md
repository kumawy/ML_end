# Experiment log — unchanged data, retrospective accuracy rounds

Date: 7 November 2026. Preparation is AI-assisted; new personal student work remains unverified until recorded in CONTRIBUTIONS.md. The sequence below distinguishes the original MAE objective, the later user-directed dual objective and evidence from a reused benchmark.

## Invariants

Source SHA-256 `b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468`; 3,999 source rows; 2,553 original clean ad IDs/targets; unchanged seed-42 split 1,787/383/383. No new/synthetic project observations, external specifications, price-derived predictors or hard-row deletion. Country/theme/group/asking-price target and required lecture controls, K-Means, PCA and ANN stay in scope. `legacy_data.py`, `reference/` and original midterm files preserve historical representations/evidence.

## 1. Earlier delivered ANN and test-informed hypotheses

The earlier endterm ANN used 147 legacy-encoded inputs, 128/64 hidden layers and its validation checkpoint at epoch 18. Its archived validation/test MAE was 2.456897m/1.754993m KZT. `ANN previous` retrains that representation/settings in the common environment: 2.457m/1.755m. Historical test errors motivated the later electric/gas–petrol, engine-state and conservative text hypotheses; the test was therefore already known.

The full representation has 15 numeric + 7 categorical inputs. Audit evidence records 16 electric, 109 structured gas–petrol, 114 normalized dual-fuel mentions including five narrative conflicts, 15 structurally inapplicable engine zeros, one positive-electric displacement conflict, two missing/invalid engine sizes and only three explicit power claims. These are source statements, not verified vehicle specifications. Extraction is row-local and never uses prices or ad-specific exceptions.

## 2. First expanded-feature search — failure remains visible

Enhanced KNN/tree/forest add OHE minimum frequencies 1/2/5. The full ANN compares the same thresholds and two widths; the original CatBoost control compares six depth/loss choices. Training CV/validation choose settings. The full ANN improves validation MAE slightly, 2.456897m → 2.446m, but worsens historical-test MAE, 1.754993m → 1.849m. This observed test failure motivated the next follow-up; the literal `ANN` row and all full-feature histories remain in the final table.

## 3. Corrected-only ablation and original MAE-selected ensemble

The next fixed families were corrected-only ANN (7 numeric + 5 categorical, OHE 2/5, hidden 64/32 or 128/64) and six blends of the validation-best single ANN with CatBoost/Random Forest at ANN weights 0.25/0.50/0.75. The corrected subset removes claims/power/trim/brand_model while retaining semantic fuel/engine fixes. It does not isolate a causal feature effect because frequency/architecture/checkpoint selection also varies.

**The original primary rule was validation MAE alone.** It selected `Ensemble` = 0.75 × ANN corrected + 0.25 × Random forest tuned: validation MAE 2,289,371.6968 KZT; historical-test MAE 1,667,212.1367 KZT, within ±20% 215/383 (56.1358%). MAE improved 5.0% versus previous ANN, but median APE worsened 16.87% → 17.25%, median absolute error 0.832m → 0.871m and within ±20% 56.92% → 56.14%. This historical tradeoff is retained, not relabelled as a dual-goal success.

## 4. Preceding steering-pool dual-goal search

The user then requested **both lower MAE and higher within-±20% fraction**. The original success criterion of at least 20% lower MAE than the training median remains unchanged; this is an added preference, not a revision of earlier selection history.

An independent TRAIN-only audit found exact structured steering clauses: 93 right-explicit and 1,694 not-stated among 1,787 training listings. The rule full-matches comma-delimited `Правый руль`/`Левый руль` clauses before standalone `КПП`. Missing, narrative-only or negated wording stays not-stated; absence is never assumed left, and opposing clauses produce conflict. No brand/model lookup or external technical data is used. The selected corrected-plus-steering view has 7 numeric + 6 categorical inputs, 307 after train-fitted OHE minimum frequency 2.

The bounded development search retained every trial: 20 CatBoost configurations × 3 training folds → six validation finalists; 12 alternative configurations × 3 folds → five finalists; eight ANN pools; 30 predeclared blends; two delivered controls. This is **51 final validation candidates**, beyond CV fits and per-epoch checkpoints. No standalone CatBoost/alternative finalist beat the delivered ensemble on both validation quantities. Protocols, scores, failed finalists, predictions and independent selection review are copied to `results/accuracy_followup/`.

Eligibility was fixed as `MAE < baseline MAE` **and** `hit fraction > baseline hit fraction`. Eligible candidates minimize `0.5 × [MAE/baseline MAE + miss fraction/baseline miss fraction]`; otherwise retain the baseline. The 30-blend grid and selected component artifacts were recorded before this round's selected-candidate historical-test evaluation. This is a local auditable record, not an immutable external preregistration.

Each pool averages all seeds 11/23/42/71/101 equally, without searching subsets. One 128 → 64 ReLU architecture, Adam lr 0.001, L2 alpha 0.001, batch 64 and training-standardized log1p target are fixed. Two checkpoints per seed minimize KZT validation MAE or the normalized dual score. Cap 300 epochs; patience 45 after neither best improves. MAE checkpoints: 56/54/72/24/21; dual checkpoints: 30/61/69/63/24; runs: 101/106/117/108/69. Validation chooses every checkpoint and the final mixture, so its gains and overfitting diagnostics are selection-dependent.

## 5. Steering-round frozen evaluation and stopped search

The frozen winner is `blend_44`, integrated as **Robust ensemble = 0.75 × ANN pool dual + 0.25 × the unchanged Random forest tuned**, mixed in original KZT. Validation MAE falls 2,289,371.6968 → 2,248,401.7334 KZT and within ±20% rises 246/383 → 254/383 (64.2298% → 66.3185%). The independent pre-test review reconstructed the predictions/weights and confirmed selection under the declared rule.

One evaluation of this frozen winner on the known historical test measured MAE 1,667,212.1367 → 1,646,174.8055 KZT (−21,037.3312 KZT; −1.2618%) and within ±20% 215/383 → 221/383 (56.1358% → 57.7023%; +1.5666 percentage points). Both requested observed metrics improved, so the candidate was retained. The fallback acceptance decision itself uses the known test and is disclosed. That bounded steering-pool search stopped; integration reproduced its frozen specification. The later separately requested dataset-enrichment work below opens a distinct follow-up, adding to the overall adaptive research history rather than restoring an unseen test.

Median absolute error improves 871,076 → 795,925 KZT and median APE 17.2489% → 16.6283%. RMSE instead worsens 3,167,634 → 3,246,983 KZT and R² falls 0.917654 → 0.913477. The lower test MAE of the standalone dual pool does not override its ineligible validation MAE. All results remain visible rather than selecting by test ranking.

Paired bootstrap: 10,000 resamples of the fixed 383 listing indices, seed 20261107. The 95% interval for MAE reduction is [−72,676; 103,502] KZT and for hit-rate gain [−1.3055; 4.4386] percentage points. Both contain zero. These are descriptive intervals conditional on this reused split/predictions; they do not cover adaptive selection bias, training variability or independent new-data performance. There is no calibrated per-car interval.

## 6. Lossless dataset enrichment and quality audit

A later dataset-focused request added source-derived fields and explicit review metadata while preserving every original observation. `dataset/used_cars_enriched.csv` has 2,553 rows and 47 columns: every prior loader column is unchanged, plus strict steering, seven candidate inputs, one age audit proxy, four quality flags and two export metadata fields. The source CSV bytes, accepted IDs/prices, order and original split are preserved. The dictionary and quality report explain roles/missingness; the 598-row review queue proposes inspection, not automatic corrections or exclusions.

TRAIN audits found 1,545/1,787 descriptions ending `...` at lengths 120–121, suggesting a capped snippet source. Strict post-КПП positional parsing finds 1,281 color labels, 642 alloy-wheel claims and 194 sunroof claims. It skips optional complete mileage and stops at unknown/negated/truncated/narrative clauses. New equipment claims are 1 or NaN: omitted wording is unknown, not absent. Lexical prefix pooling matches 254 TRAIN rows and yields 384 labels without declaring catalogue/mechanical equivalence or replacing original model labels. Age/annual-mileage are fixed-2025 mathematical proxies. All four quality flags remain audit-only; original suspicious/missing mileage and unreliable drive labels are retained.

## 7. Descriptive TRAIN-only utility study — negatives retained

Five fixed groups, three shuffled outer TRAIN folds, a 20% inner checkpoint set, one seed-42 128→64 ANN, max 150 epochs and patience 25 produced 15 fits. Imputation/missingness indicators/scaling/OHE/target scaling fit inner training only; the inner set selects MAE checkpoint and outer rows receive one held-out prediction. The following aggregates cover only the original 1,787 TRAIN rows and are not directly comparable with main validation/test five-seed results.

| Utility group | Held-out TRAIN MAE, KZT | Within ±20% | Folds with both measures improved |
|---|---:|---:|---:|
| Base corrected+steering | 2,074,610 | 57.1349% | Reference |
| Plus age/usage pair | 2,029,915 | 59.4852% | 3/3 |
| Plus lexical family | 2,120,488 | 57.1908% | 0/3 |
| Plus literal claims | 2,215,270 | 56.5193% | 0/3 |
| Plus all seven | 2,237,871 | 56.5193% | 0/3 |

Age/usage improves both quantities in all three folds; the other three groups worsen MAE in all three. No effect is asserted causal, guaranteed or independent. The original utility script's scope explicitly excluded final-model promotion or OOF-driven tuning; all five outcomes and protocols are retained.

## 8. Separate bounded age/usage validation follow-up

After the descriptive study, a separately scoped pass used only `log_vehicle_age` and `log_annual_mileage_proxy`, informed by the TRAIN results. This is an added adaptive feature-family choice, not a precommitted promotion within the utility script. No lexical/claim/all-seven group was tried on validation. The same five seeds 11/23/42/71/101, 128→64 architecture, cap 300/patience 45 and two checkpoint criteria produce 309 encoded inputs. MAE checkpoint epochs are 34/56/93/56/31; dual epochs 29/58/74/58/20; executed epochs 79/103/138/103/76.

The 15 validation candidates are the preceding Robust ensemble baseline, two new checkpoint pools and 12 blends of either pool with the baseline or unchanged forest at weights 0.25/0.50/0.75. Eligibility strictly improves both validation MAE and within20 against the prior Robust ensemble, then minimizes equal normalized MAE/miss score. The frozen winner is 0.25 × age/usage MAE pool + 0.75 × prior Robust ensemble: validation MAE 2,248,401.7334 → 2,238,371.6080 KZT; within ±20% 254/383 → 259/383 (66.3185% → 67.6240%).

One selected-winner evaluation on the known historical test then measured MAE 1,646,174.8055 → 1,634,563.5634 KZT (−0.70535%) and within ±20% 221/383 → 224/383 (+0.78329 percentage points). RMSE improves 3,246,982.8582 → 3,189,732.2921 KZT and R² 0.913477 → 0.916501. Median absolute error worsens 795,924.5401 → 801,611.0973 KZT; within ±10% and ±30% each lose one hit. No additional search follows this result. The frozen decision and historical evaluation remain development evidence; the main run verifies the same specification.

The paired-bootstrap 95% intervals for improvement are [−15,173; 41,640] KZT MAE reduction and [−0.7833; 2.6110] percentage points within20 gain. Both include zero and omit adaptive selection bias/training variability. These are descriptive fixed-prediction comparisons on an already known listing benchmark, not independent new-data confirmation or calibrated per-car intervals.

## 9. Complete executed comparison

All preceding 15 comparison rows remain; only `Usage ensemble` is appended, giving 16 rows. Its two new usage pools/12 blend alternatives remain validation tuning records, not additional historical-test rows. Routine training reproduces those bounded families and the required controls, but does not rerun the descriptive utility study, the earlier 51-candidate search or its 32 new CV configurations. Frozen evaluation/bootstrap records are preserved evidence, not fresh calculations on every run.

| Model | Representation | Val MAE, m KZT | Val ±20% | Test MAE, m KZT | Test RMSE, m KZT | Test R² | Median APE | Test ±20% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Median baseline | legacy | 6.497 | 16.4% | 5.951 | 11.301 | -0.048 | 53.1% | 17.8% |
| Linear regression | legacy | 4.647 | 32.9% | 3.499 | 6.465 | 0.657 | 35.1% | 37.6% |
| KNN midterm | legacy | 3.179 | 50.9% | 2.443 | 5.889 | 0.715 | 18.1% | 54.3% |
| Tree midterm | legacy | 3.436 | 46.7% | 2.459 | 5.485 | 0.753 | 20.8% | 48.3% |
| KNN tuned | enhanced | 3.410 | 52.7% | 2.786 | 6.416 | 0.662 | 21.0% | 47.8% |
| Tree tuned | enhanced | 3.364 | 45.7% | 2.535 | 6.207 | 0.684 | 19.1% | 52.5% |
| Random forest tuned | enhanced | 2.697 | 55.1% | 2.068 | 4.968 | 0.797 | 15.7% | 56.7% |
| ANN previous | legacy | 2.457 | 60.1% | 1.755 | 3.483 | 0.900 | 16.9% | 56.9% |
| ANN | full enhanced | 2.446 | 61.6% | 1.849 | 3.878 | 0.877 | 16.6% | 55.6% |
| ANN corrected | corrected only | 2.371 | 61.6% | 1.736 | 3.135 | 0.919 | 19.5% | 50.9% |
| CatBoost | enhanced | 2.690 | 58.0% | 1.962 | 5.173 | 0.780 | 15.7% | 58.2% |
| Ensemble | mixed | 2.289 | 64.2% | 1.667 | 3.168 | 0.918 | 17.2% | 56.1% |
| ANN pool MAE | corrected+steering | 2.274 | 65.3% | 1.653 | 3.133 | 0.919 | 16.4% | 55.9% |
| ANN pool dual | corrected+steering | 2.299 | 64.8% | 1.640 | 3.060 | 0.923 | 16.9% | 56.7% |
| Robust ensemble | mixed | 2.248 | 66.3% | 1.646 | 3.247 | 0.913 | 16.6% | 57.7% |
| Usage ensemble **(selected)** | mixed | 2.238 | 67.6% | 1.635 | 3.190 | 0.917 | 16.6% | 58.5% |

The selected test MAE is 72.53% below median, 33.09% below re-run midterm KNN and 6.86% below previous ANN. The unchanged ≥20% baseline criterion is met. The saved graph is five new 309-input MAE networks at weight 0.05 each, five retained 307-input dual networks at 0.1125 each and unchanged forest at 0.1875. Top-level `ann`/`ann_history.*` still describes the historical single-seed corrected control; `robust_ann` and `usage_ann` have separate histories/reports.

Current selected-model MAE by train-price quartile: Q1 0.783m (n=112), Q2 1.030m (n=100), Q3 1.057m (n=87), Q4 4.088m (n=84). Train cutoffs 3.6/6.7/11.5m are evaluation-only. Largest error is the 2022 EQS SUV: actual 58.000m, prediction 30.208m, absolute error 27.792m KZT. No valid high-price row was deleted.

## 10. Required unsupervised components

K-Means/PCA remain on 1,775 non-electric training rows, excluding 12 explicit electric training listings only from displacement geometry. All 1,787 remain in prediction training. Compare k=2..8 with 20 initializations, seed 42; select k=4 by silhouette 0.3686. PCA retains 80.13% in two components. Profiles, counts, loadings and plots remain. This is not an identical segmentation of the previous all-train subset and does not establish universal market groups.

## 11. Reproduction and limits

The completed current run took **122.9 seconds**, exported the 2,553×47 derived table/598-row review queue and generated all **16 comparison rows**. The selected metrics reproduce the frozen Usage ensemble. All **58 tests passed**; earlier counts of 35 and 54 belong to the steering and dataset-only stages. Tests cover source-cell preservation, deterministic fixed-year proxies, exact source clauses/unknown semantics, audit-only exclusions, both pool checkpoints/allowed additional inputs, averaging all members and strict dual selection with baseline/reference mixtures. README's saved inference derives both steering and dataset fields before saved-contract feature selection.

Python 3.12.14 and recorded packages remain numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0 and catboost 1.2.10. A prior clean install of the same 22-package lock reproduced the earlier 12 controls within 1e−6 KZT in 114.5 seconds; this is historical evidence, not a fresh clean-install claim for the current extension.

`results/dataset_enrichment_study/` retains source/quality audits, the utility protocol/all outcomes and the usage validation/frozen evaluation. Its paired bootstrap uses 10,000 NumPy default_rng resamples of paired fixed predictions, seed 20261107, percentile 2.5/97.5. The main entry point does not rerun this bootstrap or the utility experiment. The wider preceding research records remain separately under `results/accuracy_followup/`.

No test labels are used for fitting/checkpointing/candidate selection in this round, while the loader parses the original source for unchanged eligibility. Historical test feedback, TRAIN-study-informed choice, repeated validation and disclosed test acceptance prevent independent generalization claims. Both usage-round intervals include zero; listing dependence, sparse premium/claim groups and unavailable verified condition/location/dates remain limits. The extra export fields are source claims, proxies or review metadata, not recovered truth. Independent later-data validation/calibration remain future work, not new observations promised for this submission. Actual student contributions/defense remain unverified.

## 12. Bounded simplicity review — 2026-11-07

The user allowed up to 3% higher global KZT MAE for simpler inference and better
affordable-car performance. The protocol and runnable source were committed as
`1275c7c` before fitting. Three fixed simpler recipes and a rebuilt historical
recipe were assessed on exactly 1,787 original TRAIN rows in five stratified
outer folds with separate inner checkpoints. Original validation/test identities
were excluded; targets, source and split were unchanged. This is development
OOF evidence, not a new independent test.

| Fixed recipe | Inference estimators | OOF MAE, KZT | Overall ±20% hits / 1,787 | Affordable ±20% hits / 905 | Affordable median APE |
|---|---:|---:|---:|---:|---:|
| single_mae | 1 | 1,959,116 | 1,070 | 453 | 20.00% |
| single_log | 1 | 1,987,068 | 1,074 | 451 | 20.06% |
| single_forest | 2 | 1,890,899 | 1,101 | 465 | 19.06% |
| historical_recipe | 11 | 1,878,399 | 1,124 | 471 | 18.77% |

Neither single ANN met the 3% MAE cap. The ANN/forest mixture met the cap
(+0.6655%) but failed overall hits, Q2 hits and affordable median APE. Q1 hits
were 204/452 versus 203/452; Q2 were 261/453 versus 268/453. No recipe met
every frozen gate. No more candidates were added and no gates were loosened.
The historical selected model remains unchanged; no development replacement
was fitted. The original 16-row historical comparison and presentation remain.

The completed run took 36.27 seconds. All 64 tests passed (six new checks).
`results/simplicity_review/` retains hashes, identities, all predictions, learning
histories, all outcomes and the decision. The independent verifier recomputes
the metrics/gates and checks the recorded checkpoint boundaries and preserved
artifacts without fitting or evaluating new candidates on old holdouts.
