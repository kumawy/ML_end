# Reproduce the TRAIN-only utility study

From `endterm_code`, run with explicit paths and a fresh output directory:

    python experiments/utility_check/run_utility_check.py --project-source . --output-dir experiments/utility_check/new_run
    python experiments/utility_check/verify_utility_check.py --output-dir experiments/utility_check/new_run

The15-fit study is a descriptive single-ANN feature comparison, not the final ensemble's validation/test metric. The runnable project's default preprocessing is unchanged from the study; a fresh run records current source hashes.

To verify the preserved historical evidence without fitting models:

    python experiments/utility_check/verify_utility_check.py --output-dir results/dataset_enrichment_study/utility_check

This check uses the included exact historical source snapshot and relocated provenance pointer. Original provenance is retained alongside it. Main training is `python run_project.py`; standalone deterministic dataset export is `python dataset_export.py`.
