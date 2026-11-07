# Independent integration review

## Verdict

**No remaining blocking code issue found.** The integrated pipeline reproduces the frozen selected specification and historical benchmark, retains the original 12 control results, and adds the two ANN pools plus the robust ensemble. No new candidate search or parameter tuning was performed during this review.

One concrete artifact-path defect was identified and fixed by the parent: the report copied to the results root originally retained paths relative to the nested `robust_ann` directory without identifying that base. Main now records `artifact_base_directory: robust_ann` in the root report and its consolidated embedded copy. I verified every referenced history, plot, pool and all ten checkpoint files resolves under that base. The module's own nested report retains correct local paths. This was a metadata fix; no fitted model or selection changed.

## Code and API assessment

Reviewed `robust_ann.py`, `structured_features.py`, both new test modules, the full `run_project.py` integration diff and the implementation/equivalence report. The new modules use project-local imports and require no new dependencies. Original data, legacy model, ANN, CatBoost, ensemble, unsupervised and dependency implementations remain byte-identical to the user's delivered versions.

`fit_robust_pool` fits predictor imputation, scaling, OHE categories and log-target standardization on TRAIN only. It accepts TRAIN and validation, not test inputs. Its five fixed seeds, architecture, two strict checkpoint criteria and shared patience logic match the frozen scratch experiment. Deep-copied best checkpoints are restored before averaging all five seeds. Additional train/validation history predictions do not mutate fitting. The helper copies input frames; steering is derived only from complete literal pre-КПП clauses, with missing/conflict semantics preserved and no price/ID/model inference.

The selector requires strict improvement in both validation MAE and within-20% frequency before ranking eligible candidates by normalized MAE/misses. The integrated nine-candidate selection is the declared selected family: baseline, two pools and six fixed ANN/RF mixtures. It reproduces the winner of the larger 51-candidate development ledger. Its first-declared tie rule differs from that ledger's secondary MAE/name tie breaks, but there is no deciding tie and this difference is explicitly documented; it does not affect this pinned-data result.

Main fits the pools and constructs the selected ensemble before extracting test predictors/targets for evaluation. It writes `selection_before_test.json` before test predictions. It never ranks models by test metrics to change the winner. The independent known-test acceptance decision from the preceding frozen evaluation remains disclosed. The original 12 model rows, K-Means and PCA are retained. Baseline fallback and either single-pool or ANN/RF selection have coherent component flattening; the current selected bundle contains five ANN components at approximately 0.15 each and the unchanged RF at 0.25.

## Independent numerical and artifact verification

The review's verification script used the staged code and saved predictions; it did not fit a model. Machine-readable evidence is saved in `INTEGRATION_REVIEW_CHECKS.json`.

| Check | Independent result |
|---|---|
| Original 12 validation prediction arrays vs user folder | Exact equality; maximum difference 0.0 KZT |
| Original 12 historical-test prediction arrays vs user folder | Exact equality; maximum difference 0.0 KZT |
| All 15 × 2 splits × 8 metrics | All 240 independently recomputed and matched comparison CSV |
| Dataset and split identities | Source SHA-256 and exact split manifest unchanged |
| Frozen selection hash after integration | Unchanged and matches frozen evaluation record |
| Selected integrated historical-test predictions vs frozen result | Within 1e-6 KZT |
| Paired-bootstrap intervals from the fixed prediction CSV | Independently reproduced; both cross zero |
| Saved bundle with only declared feature columns | Reproduced saved historical-test predictions exactly in this run |
| Selected bundle weights/components | Six components: five × 0.15, one RF × 0.25 |
| History seed run lengths | 101, 106, 117, 108, 69 |
| Both checkpoint epochs for every seed | Match history minima and member report |
| Nested and root report artifact paths | Histories/plot/two pools/ten checkpoints resolve |

The saved mixed-bundle contract works: derive steering on the enhanced frame, select each view's declared saved features, retain row alignment, and call the saved estimator with enhanced/legacy views. Price, price_raw, ad_id and link are absent from the selected predictor lists. No transformer is refitted during inference. Both a legacy view and an enhanced view remain documented even though the current selected six components all use enhanced inputs; this preserves compatibility with the baseline fallback.

The fresh root test log records **35 tests passed in 1.757 seconds**. I inspected the six new behavioral tests: they cover literal extraction boundaries/input preservation, rejection of each one-sided gain, joint-score ranking, invalid/aligned inputs, TRAIN-only transformations/vocabulary, restored checkpoints, serialization, finite histories and image output. The implementation's full-protocol equivalence evidence reports exact validation equality for both pools; this review independently checked the final saved inference and frozen-result agreement rather than repeating training.

The pool history PNG was visually inspected: train/validation transformed losses, validation MAE, within-20% curves and checkpoint markers are readable and correctly labelled. Top-level `ann_history.*` remains the historical single-ANN control; the pool has separate histories and reports.

## Metrics and claim limits

The frozen historical comparison is reproduced: MAE 1,667,212.1367 → 1,646,174.8055 KZT and within ±20% 215/383 → 221/383. RMSE worsens 3,167,634.1132 → 3,246,982.8582 KZT; R² decreases 0.917654 → 0.913477. Thus the two user-requested observed measures improve, but improvement is not universal.

README, METHODOLOGY and EXPERIMENT_LOG now describe the 15 rows, the strict-both rule, selected-family integration versus the wider development search, repeated checkpoint/validation selection, historical-test reuse, adverse RMSE/R², and both descriptive bootstrap intervals crossing zero. They distinguish empirical within-20% frequency from a confidence level and distinguish the current 35-test run from an earlier clean installation of the original 12 controls. The full CSV loader's eligibility pass is correctly distinguished from test-label use for fitting/selection. The historical MAE-only chronology is now explicitly scoped to the earlier pass.

After the documentation author finished, I independently re-read all six final documents, including the Russian notes, compliance map and contribution disclosure. Both 15-row Markdown tables match all 210 displayed numeric values after the documented rounding. The 15 Russian sections total 700 seconds, and their seed-42 checkpoint loss statement matches the saved history. The notes use the final selected metrics/error bands, disclose adverse RMSE/R² and zero-spanning intervals, and do not turn proposed student responsibilities into completed work. No documentation consistency issue remains open in this review. The deck itself is being reviewed by its author/parent rather than certified by this code review.

The remaining substantive limitation is statistical, already disclosed: the benchmark and validation are reused; 51 development predictions exclude additional CV/checkpoint choices; paired intervals omit training and adaptive-selection uncertainty. The measured changes support a descriptive improvement on this historical split, not independent new-data generalization or a causal steering effect.

No production code was modified by this reviewer.
