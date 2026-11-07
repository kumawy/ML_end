# ML_end

Machine Learning Algorithms endterm project: estimating used-car **advertised asking prices in Kazakhstan**.

The project continues the original midterm dataset and preserves its cleaned advertisements, targets and train/validation/test assignments. It includes classical regression models, K-Means, PCA, feedforward neural networks, CatBoost, fixed neural-network ensembles and an auditable enriched dataset.

## Project files

- [Complete code, setup and methodology](endterm_code/README.md)
- [Presentation](ML_Endterm_Presentation.pptx) · [PDF](ML_Endterm_Presentation.pdf)
- [Enriched dataset](endterm_code/dataset/used_cars_enriched.csv) · [Data dictionary](endterm_code/dataset/data_dictionary.csv)
- [Model comparison](endterm_code/results/model_comparison.csv)
- [Experiment history](endterm_code/EXPERIMENT_LOG.md)
- [Course requirement mapping](endterm_code/REGULATION_COMPLIANCE.md)
- [Contribution records and assistance disclosure](endterm_code/CONTRIBUTIONS.md)
- [Simplicity and affordable-car review](endterm_code/SIMPLICITY_REVIEW_RU.md)

## Run

Use Python 3.12. From the repository root:

```bash
cd endterm_code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_project.py
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`. The recorded dependency lock and environment are included in `endterm_code/`. Training uses the bundled source CSV and downloads no data. The saved model, learning histories, comparison tables and original split manifest are included.

To regenerate only the derived dataset:

```bash
python dataset_export.py
```

## Recorded results

The retained dataset has **2,553 unique used-car advertisements**, split into **1,787 training / 383 validation / 383 historical-test rows**. The derived export contains 47 columns, including audit metadata; only declared input fields enter the models.

The validation-selected `Usage ensemble` has the following recorded historical-test metrics:

| Metric | Result |
|---|---:|
| MAE | 1,634,564 KZT |
| RMSE | 3,189,732 KZT |
| R² | 0.9165 |
| Predictions within ±20% | 224/383, or 58.49% |

All 16 model rows and additional error measures are retained. R² is not a percentage of accurate individual predictions. The historical test has informed earlier development, and validation has been reused; these results are a retrospective benchmark, not independent confirmation on new market data. The latest paired improvement intervals include zero. See the full methodology and experiment log for mixed metric changes and limitations.

The project includes 64 behavioral and regression tests. Source descriptions are often truncated; literal equipment claims are not verified vehicle condition, and missing equipment remains unknown. The age/usage fields are fixed-2025 mathematical proxies. No original price, mileage or difficult evaluation row was altered to improve the score.

An optional [TRAIN-only simplicity experiment](endterm_code/experiments/simplicity_review/README.md) compared four fixed recipes under a user-approved 3% global-MAE tolerance and explicit affordable-car checks. None of the three simpler recipes met every condition: the closest two-component mixture increased MAE by 0.67%, but lost 23/1,787 overall ±20% hits and 6/905 affordable-car hits. Its out-of-fold results are development evidence, separate from the historical test table above. The original selected model and course-project pipeline are preserved.

## Data provenance

Source: [Kolesa-Cars-2025 on Kaggle](https://www.kaggle.com/datasets/mutashevdarkhan/kolesa-cars-2025). Original midterm: [AlisherAt/MLMid](https://github.com/AlisherAt/MLMid).

Source hashes, uploader-declared licensing metadata and provenance qualifications are documented in the code README. This publication does not assert additional rights over third-party listing content. Historical development records may contain original local artifact paths; those paths are provenance, not runtime dependencies. The included source snapshots and portable verification commands allow the packaged evidence to be checked without those private research binaries.
