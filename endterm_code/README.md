# Used-car price estimation in Kazakhstan — improved endterm

Machine Learning Algorithms. Original team: **Alisher Akhmet and Aslan Muratov**.

## Question and success criterion

Estimate the **advertised asking price** of a used car in Kazakhstan from its existing vehicle specifications and conservative description-derived attributes. This is regression; target `price` is in KZT. **MAE in original tenge** was the original primary selection metric. The retained midterm success criterion is test MAE at least 20% below a training-median predictor. In the latest follow-up, the user additionally required both lower MAE and a higher fraction within ±20%; candidates must improve both validation metrics, then minimize equally weighted normalized MAE and miss fraction. RMSE, R², median absolute percentage error (median APE), and fractions within ±10/20/30% remain visible.

The original country, theme, group, source CSV, 2,553 clean listings, target values and historical split are retained. The package includes midterm controls, tuned supervised models, K-Means, PCA, ANN, native-categorical CatBoost, fixed five-seed ANN pools and a lossless derived dataset with explicit source-quality metadata. K-Means is exploratory. For this tabular dataset the feedforward ANN satisfies the required deep-learning component; no CNN/NLP/RL extension is forced onto it.

## Quick start

Use **Python 3.12**. In this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_project.py
python -m unittest discover -s tests -v
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`. Remaining commands are the same. Script paths resolve relative to the code directory. Training reads the local original CSV and downloads no data. Runtime depends on the computer and includes all retained controls. `requirements.txt` pins the core runtime; `requirements-lock.txt` records the verified dependency set. See `results/environment.json` for Python/OS and actual versions.

## Files and execution order

1. `run_project.py` is the top-to-bottom `# %%` project entry point. It performs audit/cleaning, fixed splitting, training EDA, K-Means/PCA, supervised searches, previous/full/corrected ANN comparisons, CatBoost, the previous blends, the fixed ANN pools, frozen dual-metric selection and retrospective test evaluation.
2. `data.py` derives corrected and enriched predictors from existing rows; `legacy_data.py` preserves the original representation for genuine unchanged controls. Learned transformations fit within training folds.
3. `ann.py`, `catboost_model.py`, `ensemble.py`, `unsupervised.py` hold the original model procedures. `structured_features.py` extracts literal steering clauses; `robust_ann.py` trains the fixed five-seed pools, supports only the declared additional age/usage inputs, and selects bounded pool/forest/reference mixtures.
4. `results/report.json` consolidates all results; `model_comparison.csv` contains all 16 rows: all preceding 15 plus `Usage ensemble`. The two new usage checkpoint pools and their blend alternatives remain validation tuning records, not extra test-comparison rows. `all_validation_predictions.csv` and `all_test_predictions.csv` preserve comparable predictions. `feature_audit.csv`, `split_manifest.csv`, search CSVs, learning histories and selected-model errors retain traceability.
5. `results/ann_history.*` show the best historical single-seed ANN (`ANN corrected`); `ann_full_minfreq_*`, `ann_corrected_minfreq_*` and `previous_ann/` retain its controls. `results/robust_ann/robust_ann_history.*` and `robust_ann_report.json` document the 307-input control pool. `results/usage_ann/robust_ann_history.*`, `usage_ann_report.json` and `usage_selection_report.json` document the 309-input follow-up. `catboost_cv.csv`, `catboost_history.*` and `ensemble_validation_search.csv` retain the original candidates; `results/accuracy_followup/` retains the earlier 51-candidate ledger. `results/dataset_enrichment_study/` retains source/quality audits, all utility-study outcomes, the 15-candidate usage ledger, frozen decisions and descriptive bootstrap evidence.
6. `reference/previous_endterm_report.json`, `reference/previous_endterm_comparison.csv`, and `reference/previous_ann_predictions.csv` preserve the earlier delivered ANN version. `midterm_reference.py` and `midterm_reference_metrics.json` preserve the historical midterm. Run the original reference script only in a separate copy: it writes its own `results/` outputs.
7. `METHODOLOGY.md`, `EXPERIMENT_LOG.md`, `DEFENSE_NOTES_RU.md`, and `CONTRIBUTIONS.md` explain interpretation, the failed first search/follow-up chronology, the 15-slide defense, and contribution records.

`dataset_features.py` adds deterministic source-derived candidates and review signals without overwriting source cells. `dataset_export.py` writes the enriched CSV, data dictionary, quality report and manual review queue under `dataset/`; the main entry point also runs this export. To reproduce only the export without training models:

```bash
python dataset_export.py
```

## Real source and provenance

Dataset: [Darkhan Mutashev, Kolesa-Cars-2025, Kaggle](https://www.kaggle.com/datasets/mutashevdarkhan/kolesa-cars-2025). Original midterm repository: [AlisherAt/MLMid](https://github.com/AlisherAt/MLMid), snapshot `059b9319ea07b40cd49e5946f4a0b31a9d4e5bce`.

The original README records MIT in Kaggle catalog metadata and 18 December 2025 as its update date. These are source metadata, not proof of each ad's date or of ownership of Kolesa content. All source prices are labelled ₸. This is a 2025 asking-price snapshot, not current 2026 transaction data. No new or synthetic project observations were added.

CSV SHA-256: `b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468`. Original Git blob SHA-1: `01e57295927f91ffd7996745ce6a867b6d41b0f7`. The loader verifies the original blob. Both loaders must produce identical source indices, ad IDs and prices; these identities are asserted before training.

## Cleaning and feature semantics

From 3,999 rows, remove 527 new cars, 51 on-order used-car rows, then 868 repeated remaining ads; no additional rows fail the retained validity checks. This leaves **2,553 unique used listings**. The 968 repeats across the entire source and 868 removed after filters use different denominators. Mileage remains missing in 497 listings. `drive` is omitted because 3,964 original rows lack it.

The improved representation recovers **16 electric** and **109 gas–petrol** specifications from the structured description prefix before `КПП`. All 114 normalized dual-fuel mentions include those structured specifications; **5** narrative claims disagree with structured petrol. Such rows keep the structured fuel and receive disagreement flags. Unknown fuel remains `unknown`, never a fabricated most-common fuel.

**15 electric listings** have 0 L with `engine_not_applicable=1`; this means displacement is inapplicable rather than missing. One electric/positive-displacement source disagreement retains the literal value with `engine_source_conflict=1`. **2 genuinely missing/invalid engine values** stay missing and get `engine_missing=1`. Car names are never used to invent displacement or power.

Full features are **15 numeric + 7 categorical = 22 raw inputs**: original predictors plus engine-state flags, fuel/text disagreement flags, explicit equipment/warranty claims, explicit power/missingness, `brand_model` and advertised trim tokens. The corrected-only ablation uses **7 numeric + 5 categorical = 12 raw inputs**: original predictors on corrected fuel/engine data plus engine-state flags. It excludes extra claims, power, trim and `brand_model` to test their contribution. The new ANN pools add one categorical steering field to this corrected subset: **7 numeric + 6 categorical = 13 raw inputs**, yielding **307 encoded inputs** with training-fitted minimum frequency 2.

Steering is accepted only from an exact comma-delimited `Правый руль` or `Левый руль` clause before a standalone `КПП` boundary in the existing description. Missing, narrative-only, negated or ambiguous wording is not inferred from brand/model. Opposing explicit clauses yield `conflict`; absence stays `not_stated`, never assumed left-hand steering. The TRAIN-only audit found **93 right-explicit and 1,694 not-stated** values among 1,787 listings; it did not establish left-hand prevalence or a causal steering effect.

The earlier broad text flags describe seller wording, not verified equipment or condition; zero means no supported positive claim. Local negation is rejected. The new exact equipment fields instead use **1 for a supported positive clause and NaN for unknown**, never a verified negative. Only 3 listings contain a recognized explicit power unit; bare numbers and asking prices do not create power. Price, price_raw, ad ID and URL never enter predictors; full free text is not fed directly to a model.

### Enriched dataset and source-quality audit

`dataset/used_cars_enriched.csv` contains **2,553 rows × 47 columns**, including provenance and split metadata. Every value in the loader's 32 existing columns, every retained ID/price and the original order/split is preserved; the raw source CSV remains byte-identical. The added columns comprise existing strict steering, seven new candidate inputs, one age audit proxy, four quality flags and two export metadata columns. `data_dictionary.csv` defines roles/missingness; `quality_report.json` records hashes/counts. `review_queue.csv` lists **598 records** with missing mileage, older/low or repeated-digit mileage, or a present unreliable original drive label; truncation alone does not enqueue a row. Review signals do not delete, repair, rescale or certify any source value.

TRAIN-only audits established the extraction rules. **1,545/1,787 descriptions (86.46%)** end in `...` at 120–121 characters, suggesting capped source snippets; the exact upstream mechanism is unknown. Missing text is unavailable evidence. The strict post-КПП parser reads an optional complete mileage clause, then an exterior label and a contiguous block of exact recognized source options, stopping at unknown, negated, truncated or narrative clauses. It finds **1,281 literal colors**, **642 alloy-wheel claims** and **194 sunroof claims** in TRAIN. These are seller labels, not verified color/equipment. A panoramic-roof clause does not automatically mean a sunroof. Multiple upholstery labels become `multiple_literals`, not a claim of physical contradiction.

`model_family_lexical` groups only narrow brand-scoped literal prefixes and retains original brand/model fields. It matches 254 TRAIN rows and yields 384 categories overall in TRAIN; it is not a vehicle-catalog family, generation or mechanical-equivalence claim. Unmatched names remain distinct. The four audit-only signals flag description truncation, low mileage in an older model-year vehicle, repeated-digit mileage and present but unreliable original drive labels. A flag is a review reason, not proof of an error. Missing and suspicious mileages remain unchanged; absent equipment is unknown.

The seven new candidates are `log_vehicle_age`, `log_annual_mileage_proxy`, `alloy_wheels_claim`, `sunroof_claim`, `model_family_lexical`, `color_literal` and `upholstery_literal`. Age is fixed to **2025 − model year**; annual usage is `mileage / max(age, 1)`, then log1p. These are source-snapshot proxies, not registration age or verified yearly driving. The code never uses the current year. The quality flags, age audit column, IDs, prices, raw text and split metadata are excluded from model predictors.

### Controlled utility study and later model follow-up

A separate descriptive TRAIN-only utility study compared five fixed groups: base, age/usage pair, lexical family, literal claims, and all seven fields. Three outer folds with a 20% inner checkpoint split, one ANN seed 42, cap 150 and patience 25 produced **15 fits**; all preprocessing/target scaling fitted only on inner training rows. Aggregate held-out TRAIN MAE was **2,074,610 → 2,029,915 KZT** for the age/usage pair, with within ±20% **57.1349% → 59.4852%**, improving both measures in all three folds. Lexical family, literal claims and all seven together worsened MAE and remain visible. These scores are not directly comparable with the main validation/test five-seed ensemble.

The study's original scope was descriptive, without promoting a final model. A later separate bounded follow-up used only the age/usage pair, with the same five-seed 128→64 pool protocol and **309 encoded inputs**. It examined 15 validation candidates: the prior robust baseline, two checkpoint pools, and 12 fixed blends with that baseline or the unchanged forest, weights 0.25/0.50/0.75. Strictly improve both validation MAE and within ±20%, then minimize equal normalized MAE/miss score. No other feature group was tried on validation. This follow-up is informed by the TRAIN study and adds an adaptive choice; it does not turn the descriptive study into independent confirmation.

Training EDA retains mean 9.53m versus median 6.70m KZT; Spearman year/price 0.69 and mileage/price −0.53. Associations are not causal. EDA uses only training listings.

### Four explicit EDA plots and their interpretations

The midterm regulation requires at least four meaningful EDA plots, each with a short interpretation. The current entry point produces all four in `results/`:

| Plot | Interpretation |
|---|---|
| `eda_prices.png` | Mean 9.53m exceeds median 6.70m KZT: high-price listings create a long right tail. |
| `eda_year.png` | Newer model years generally have higher median asking prices; the association does not prove causality. Displayed years have at least eight training listings. |
| `eda_mileage.png` | Higher observed mileage is associated with lower asking price, with substantial spread. The plot uses up to 1,200 known-mileage training rows sampled with seed 42. |
| `eda_fuel.png` | Fuel groups differ in observed price distributions, but also in brand/year mix. The plot retains every training row, uses a labelled logarithmic price axis and shows group counts. Electric (n=12) and hybrid (n=22) groups are small. |

The same four interpretations and file names are saved in `results/report.json` → `eda.plots`. The fuel-group CSV is supporting evidence, not a replacement for the fourth chart.

### Scope fixed by the two course regulations

See `REGULATION_COMPLIANCE.md` for the requirement-to-evidence map. The final submission continues the original dataset/theme/group and retains the lecture baselines, K-Means, PCA and ANN. CatBoost and the ensemble are additional comparisons. The current endterm presentation follows **10–15 slides / 12 minutes**; the midterm's **7–12 slides / 10 minutes** applies to its historical defense. New dated data are optional external validation, not a replacement for the original endterm dataset. Actual student contributions and defense participation still need personal evidence.

## Validation and honest test interpretation

Preserve **1,787 train / 383 validation / 383 test**, seed 42, after ad-ID deduplication. Supervised model searches use one shuffled 3-fold KFold within train; their one-hot minimum-frequency candidates are 1/2/5. ANN compares full features at 1/2/5 and corrected-only at 2/5, with two architectures/checkpoints per representation. Numeric imputation/scaling, categorical vocabularies/grouping, and ANN target standardization learn only from the appropriate training rows. CatBoost keeps native categories and numeric missingness. All final estimator fits use training observations; validation controls checkpoint/configuration selection, and no train+validation refit changes the comparison.

The first improvement pass used six fixed blends of the validation-best single ANN with CatBoost or Random Forest, with ANN weights 0.25/0.50/0.75. **Validation MAE alone selected the delivered `Ensemble` in that pass.** Its 12 rows and all failed changes remain controls.

The preceding steering-pool accuracy round is separately recorded: 20 CatBoost configurations × 3 training folds produced six validation finalists; 12 alternative configurations × 3 folds produced five finalists; eight fixed ANN pools and 30 declared blends were also evaluated. With the two delivered controls this is **51 final validation candidates**, excluding the additional CV fits and per-epoch checkpoint choices. No standalone CatBoost/alternative finalist beat the delivered ensemble on both required validation metrics. Those records remain under `results/accuracy_followup/`. Its nine production follow-up candidates reproduce the prior `Robust ensemble`; the latest bounded age/usage pass is a separate 15-candidate validation study. Neither the wider 51-candidate search nor the TRAIN utility study is rerun by routine final training; their evidence and negative results are retained.

Let `h` be the fraction within ±20% and let subscript `b` denote **that round's retained reference**: `Ensemble` for the steering round, then `Robust ensemble` for the age/usage round. A candidate is eligible only if `MAE < MAE_b` and `h > h_b`. Eligible candidates minimize `0.5 × [MAE / MAE_b + (1−h) / (1−h_b)]`; otherwise keep that reference. Each five-seed checkpoint pool averages every predeclared seed equally; no seed subset is searched. Each winner's specification/artifacts were frozen before its selected-candidate evaluation. The latest bounded search stopped after that evaluation; the full project rerun verifies the same frozen specification.

**This is a test-informed retrospective benchmark.** Known midterm/previous-endterm test errors motivated feature hypotheses. The first full-feature search slightly improved validation MAE but worsened historical test MAE; that outcome motivated the corrected-only ablation and first blends. The latest user-directed dual objective also follows known historical outcomes. No test labels were used for fitting, checkpointing or candidate selection in this round. The unchanged common loader still reads the original full CSV to apply eligibility before splitting, so literal claims that test labels were never accessed would be inaccurate. Retaining the baseline if the frozen candidate had failed either historical-test goal was also a disclosed test-based deployment decision. Freezing the round does not remove earlier test feedback or repeated validation selection. See `EXPERIMENT_LOG.md`; later dated external data are needed for independent evidence.

## Actual final results

The completed run preserves all preceding 15 model rows and appends only `Usage ensemble`, on identical 383 validation and 383 historical-test ads. New usage-pool variants stay in validation tuning records. Values below come from `results/model_comparison.csv`; percentages are rounded.

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

Selected model: **Usage ensemble = 0.25 × age/usage MAE-checkpoint pool + 0.75 × prior Robust ensemble**, averaged in original KZT. Its validation MAE is **2,238,372 KZT**, with **259/383 = 67.6240%** within ±20%, versus 2,248,402 KZT and 254/383 for prior Robust. Historical-test MAE is **1,634,564 KZT**, with **224/383 = 58.4856%** within ±20%. Test MAE is 72.53% below median, 33.09% below re-run midterm KNN and 6.86% below the previous ANN. The retained ≥20% baseline criterion remains met.

Against the **prior `Robust ensemble` immediately before this dataset round**, test MAE falls **1,646,175 → 1,634,564 KZT (−11,611 KZT; −0.70535%)** and within ±20% rises **221 → 224 of 383 (+0.78329 percentage points)**. RMSE improves **3,246,983 → 3,189,732 KZT** and R² **0.913477 → 0.916501**. Median APE improves slightly **16.6283% → 16.6088%**, but **median absolute error worsens 795,925 → 801,611 KZT**, and within ±10% and ±30% each lose one hit (132→131 and 287→286). Both requested measures improve on this historical split; not every summary improves.

The paired fixed-listing bootstrap uses 10,000 draws, seed 20261107 and percentile intervals: **[−15,173; +41,640] KZT for MAE reduction** and **[−0.7833; +2.6110] percentage points for within-±20% gain**. Both include zero. These are descriptive intervals conditional on the fixed predictions and reused 383-row split; they do not account for adaptive selection or establish independent new-data improvement. They are not per-car intervals. The frozen record is preserved at `results/dataset_enrichment_study/usage_validation/frozen_evaluation.json`; routine main training does not rerun the bootstrap or the descriptive utility study.

The selected saved graph has **ten ANN components and the unchanged forest**: five new **309-input MAE-checkpoint networks at weight 0.05 each**, five prior **307-input dual-checkpoint networks at 0.1125 each**, and **Random forest tuned at 0.1875**. Every network uses the fixed seeds 11/23/42/71/101, hidden 128→64 ReLU, one linear output, Adam lr 0.001, batch 64, L2 alpha 0.001 and training-standardized log1p KZT. Each pool first averages all five seeds equally; no subset or per-row gate is searched. The two extra numeric inputs are the fixed-2025 log age and annual-mileage proxies; literal equipment/color/family and all quality flags are excluded from the selected graph.

New 309-input MAE checkpoints are **34/56/93/56/31**; dual checkpoints **29/58/74/58/20**; executed runs **79/103/138/103/76**, cap 300 and patience 45 after neither best improves. The retained 307-input dual checkpoints remain **30/61/69/63/24**, runs **101/106/117/108/69**. `ANN previous`, full 22-feature `ANN`, `ANN corrected`, CatBoost and both earlier ensembles remain visible, including failed changes. The original MAE-only pass and preceding dual-goal pass retain their own chronology/results in `EXPERIMENT_LOG.md`; the age/usage follow-up does not retroactively change them.

Top-level report `ann` and `ann_history.*` remain the historical single-seed `ANN corrected` control (305 inputs, epoch 71/111). `robust_ann` is the 307-input pool, and `usage_ann` is the 309-input pool. These reports are not interchangeable. The original CatBoost control remains checkpoint 993/1000. The archived midterm KNN MAE was 2,444,769 KZT; common-environment re-runs do not claim bitwise identity with archival metrics.

## K-Means and PCA

Clustering/PCA deliberately use **1775 non-electric training listings**, excluding 12 explicit electric training records because displacement is inapplicable. Electric listings remain in all predictive fits/evaluations. Train median imputation and StandardScaler transform year, log mileage and engine size. Neither price nor categories enters the geometry.

Compare k=2..8, 20 initializations, seed 42; highest training silhouette selects **k=4**, silhouette **0.3686**. Cluster profiles use observed medians/counts and post-hoc prices. PCA variance shares are **47.67%, 32.46%, 19.87%**; first two retain **80.13%**. It visualizes structure and does not enter prediction. This intentionally different subset must not be compared as an identical segmentation of all previous 1,787 training rows. No multi-seed cluster stability claim is made.

## Selected-model errors and practical limits

Test MAE by training-price band: **Q1 0.783m (n=112); Q2 1.030m (n=100); Q3 1.057m (n=87); Q4 4.088m (n=84)**. Quartile boundaries come from training prices at 3.6/6.7/11.5m KZT; bands use true prices for retrospective evaluation, not as predictor inputs. Fuel/model-support/age/missing-mileage tables retain sample counts. Median APE is the median of `abs(actual−predicted)/actual`; within ±20% is the share of rows with that ratio at most 0.20, not a confidence interval.

The validation tail includes 222m KZT, while test max is 93.5m with no listing above 100m. Composition helps explain score differences. The source has no individual observation date, city, verified sale price or structured detailed technical condition; `condition` only means new/used. Descriptions are truncated/unverified, rare claims are sparse, and fuel corrections do not prove causality. A common test split or higher score does not establish current-market fairness, calibration or deployment reliability. Use estimates as exploratory assistance with human inspection, not automatic transaction or credit decisions. No calibrated per-car prediction interval is claimed; the paired comparison bootstrap above has a different purpose. New dated external validation is future work, not promised new rows in this submission.

## Saved model and inference contract

`results/selected_model.joblib` includes the selected name, representation, feature contract, estimator, and separate ANN preprocessor where required. Retain its Python modules/runtime. A mixed bundle consumes **two aligned DataFrames**: `{'legacy': X_legacy, 'enhanced': X_enhanced}`; each component then uses its saved representation/preprocessor. Rows must refer to the same cars in the same order. Its enhanced contract includes `steering`, `log_vehicle_age` and `log_annual_mileage_proxy`. Apply both derivation helpers to the original loaded frame before column selection; calling `add_dataset_features` again on an already enriched frame deliberately rejects collisions. The following reproduces inference on the original historical-test rows from this code directory:

```python
from pathlib import Path
import joblib
import data
import legacy_data
from structured_features import add_structured_features
from dataset_features import add_dataset_features

root = Path.cwd()
bundle = joblib.load(root / 'results' / 'selected_model.joblib')
enhanced, _ = data.load_data(root)
legacy, _ = legacy_data.load_data(root)
enhanced = add_structured_features(enhanced)
enhanced = add_dataset_features(enhanced)
rows = data.split_data(enhanced)['test']
assert enhanced.index.equals(legacy.index)
assert enhanced.ad_id.equals(legacy.ad_id)
frames = {'legacy': legacy.loc[rows], 'enhanced': enhanced.loc[rows]}
views = {name: frames[name].loc[:, columns]
         for name, columns in bundle['features'].items()}
predictions_kzt = bundle['estimator'].predict(views)
```

This recipe applies to the selected mixed bundle. An individual model uses its saved feature list and, when non-null, its saved `preprocessor` before `estimator.predict`. Do not fit a new transformer during inference. The snapshot loader verifies the original CSV and is not a general loader for altered or future data. A future external set needs a separate preparation/provenance protocol with the same literal feature semantics.

## Contribution and reproducibility status

Seed 42, exact split IDs, CSV hashes, search candidates, checkpoints and environment are recorded. Recorded runtime: Python 3.12.14; numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, catboost 1.2.10. Hardware/BLAS differences can change numerical details.

The completed current run took **122.9 seconds**, produced **16 model rows**, retained the source CSV and exact split, and reproduced the frozen Usage ensemble result. All **58 tests passed**. Earlier 35-test steering and 54-test dataset-only counts describe prior stages, not the current total. A prior clean-environment check of the unchanged 22-package lock reproduced the earlier 12 controls (KZT metrics within 1e−6) with byte-identical CSV/split in 114.5 seconds; that historical check is not presented as a new clean installation of the current extension. The derived export fits no learned transformation; model fitting/scaling remains training-only.

To verify the preserved TRAIN utility evidence without fitting networks:

```bash
python experiments/utility_check/verify_utility_check.py --output-dir results/dataset_enrichment_study/utility_check
```

Its provenance points to a hashed historical source snapshot because the pool API was later extended; original provenance/verification records are also retained. For a separate rerun, use `python experiments/utility_check/run_utility_check.py --project-source . --output-dir experiments/utility_check/new_run`. This reruns the descriptive TRAIN study into a new folder; it is outside routine final training and adds no independent evaluation data.

Endterm code, experiments, prose and deck were prepared with Codex assistance. Historical student roles are retained; new personal technical review/edits and defense participation are **not verified**. Each student must perform and record their real contribution in `CONTRIBUTIONS.md` before submission. No AI preparation record substitutes for that requirement.
