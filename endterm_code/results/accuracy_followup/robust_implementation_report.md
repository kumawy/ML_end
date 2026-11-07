# Robust ANN implementation and verification

Changes are isolated to `/Users/muratovaslan/Documents/Codex/2026-11-07/z/work/accuracy_round2/code`. This subtask did not edit the user's source folder, the project CSV, `run_project.py`, or project documentation. The parent agent owns integration, publication and the already frozen historical-test assessment. This subtask evaluated TRAIN/validation only.

## Added files and API

- `code/structured_features.py`: `extract_structured_steering(description)` and `add_structured_features(frame)`. The helper copies its input and reads only exact comma-delimited `правый руль` / `левый руль` clauses before a standalone КПП. It handles conflicts and missing statements explicitly and never infers steering from identity, price or brand/model.
- `code/robust_ann.py`: `fit_robust_pool(train_frame, y_train, validation_frame, y_val, baseline_predictions, output_dir, seed=42, *, _config=None) -> (models, report)`. Models are existing `PredictionEnsemble` instances under `ANN pool MAE` and `ANN pool dual`. Inputs are copied before steering extraction. Default configuration is an immutable dataclass: five seeds `(11,23,42,71,101)`, one `(128,64)` ReLU/Adam network, alpha `.001`, batch 64, learning rate `.001`, at most 300 epochs, and patience 45 after neither checkpoint has improved. The private config override is used only for tiny real-training tests.
- `select_dual_blend(validation_predictions, y_val, baseline_name, ann_names, forest_name, weights=(.25,.5,.75)) -> report`. It considers exactly the baseline, the two ANN pools and the six fixed ANN/RF convex blends under defaults. A candidate is eligible only when validation MAE strictly decreases AND validation within-20% strictly increases. Eligible candidates minimize `.5*(MAE/baselineMAE + missFraction/baselineMissFraction)`; absent any eligible result the baseline is retained.
- `make_robust_preprocessor(min_frequency=2)` creates the fresh corrected numeric/categorical plus steering transformer. All imputation, standardization, category grouping and log-target standardization fit TRAIN only. It is independent of mutable feature-list changes in the scratch experiment.
- `code/tests/test_structured_features.py` and `code/tests/test_robust_ann.py`: six new behavioral tests.
- `verify_robust_equivalence.py`: TRAIN/validation-only full-configuration equivalence check against the frozen scratch artifacts.

Inference contract: `pool.predict({'enhanced': add_structured_features(frame)})`. Callers may also provide an enhanced frame that already contains steering. Omitting steering fails instead of silently deriving a different representation. The helper is re-exported from `robust_ann` for convenience.

The normalized score requires baseline MAE and baseline miss fraction to be positive; a perfect baseline is rejected with a clear ValueError rather than producing undefined normalized ratios. The delivered baseline satisfies this condition.

## TDD evidence

Read `test-driven-development/SKILL.md` and `writing-good-tests.md` before implementation. Added the six tests before either production module existed.

RED commands, both run from staged `code`:

```text
/Users/muratovaslan/Documents/Codex/2026-11-07/z/work/endterm_improvement/venv/bin/python -m unittest discover -s tests -p 'test_*structured_features.py' -v
Ran 2 tests: FAILED (failures=2)

/Users/muratovaslan/Documents/Codex/2026-11-07/z/work/endterm_improvement/venv/bin/python -m unittest discover -s tests -p 'test_robust_ann.py' -v
Ran 4 tests: FAILED (failures=4)
```

Every RED failure was the expected assertion that the corresponding module was missing, rather than an import error or typo. Then implemented the modules. The first GREEN attempt passed steering and selection tests but an unrealistic training fixture (`year=999999`, `log_km=888888`) caused floating-point mean overflow and an unset checkpoint error in `test_training_preserves_train_only_transforms_checkpoints_and_serialization`. Corrected the fixture to out-of-training-range but meaningful `year=2010`, `log_km=9`; default training math and hyperparameters were not changed. Final targeted runs passed 2/2 and 4/4 with no warnings.

Behavior protected:

- Full-clause, pre-КПП boundaries; repeated/conflicting/missing steering; case and whitespace; negation/narrative rejection.
- Input copy preservation and no price/identity/brand influence on steering.
- Strict rejection of either one-sided metric improvement and baseline retention when none qualify.
- Ranking by the requested normalized dual objective rather than MAE alone; finite/positive target, aligned prediction and valid weight checks.
- Real tiny training with TRAIN-only numeric imputation, category vocabulary and target scaling; best checkpoint restoration for both criteria; enhanced representation and missing-steering failure.
- Finite, nonnegative predictions, exact joblib round trip, JSON-safe reports, complete finite history values and valid PNG output.

Full project command:

```text
/Users/muratovaslan/Documents/Codex/2026-11-07/z/work/endterm_improvement/venv/bin/python -m unittest discover -s tests -v
Ran 35 tests in 1.743s
OK
```

No test failures remain. The full suite has 29 existing tests plus the six new tests.

## Exact scratch equivalence

Command:

```text
/Users/muratovaslan/Documents/Codex/2026-11-07/z/work/endterm_improvement/venv/bin/python /Users/muratovaslan/Documents/Codex/2026-11-07/z/work/accuracy_round2/verify_robust_equivalence.py
```

Completed with exit 0. Evidence: `robust_verification/exact_equivalence.json`, `robust_verification/robust_ann_report.json` and the serialized pools/checkpoints.

| Check | Result |
|---|---|
| Encoded TRAIN input columns | 307 |
| Validation predictions versus frozen scratch MAE pool | Exact array equality; max difference 0.0 KZT |
| Validation predictions versus frozen scratch dual pool | Exact array equality; max difference 0.0 KZT |
| MAE best epochs, seeds 11/23/42/71/101 | 56, 54, 72, 24, 21; exact scratch match |
| Dual best epochs, seeds 11/23/42/71/101 | 30, 61, 69, 63, 24; exact scratch match |
| Epochs run | 101, 106, 117, 108, 69 |
| Selected components | ANN pool dual + Random forest tuned |
| Selected weights | 0.75, 0.25 |
| Selected validation MAE | 2,248,401.7333698105 KZT |
| Selected validation within ±20% | 0.6631853785900783 |
| Historical test evaluated by this subtask | No |

The per-seed fitting loop runs with one thread, matching scratch. Additional TRAIN/validation predictions used to record descriptive losses do not mutate the estimator or alter checkpoint choices; full equality confirms this on the frozen experiment.

## Artifacts and interpretation

`fit_robust_pool` saves `robust_ann_history.csv`, `robust_ann_history.png`, two pool joblibs, ten selected per-seed checkpoint components under `robust_ann_checkpoints`, and `robust_ann_report.json`. The history includes epoch, seed, TRAIN/validation standardized-log MSE, TRAIN/validation KZT MAE, validation within-20%, dual score and checkpoint improvement flags. The report includes all fixed configuration, 307-column count, members' selected epochs and run lengths, candidate validation metrics, selected checkpoint loss/gaps and post-checkpoint changes, data-source semantics, artifact paths and explicit no-test / selection-dependence caveats.

For all five selected dual checkpoints, the last validation log MSE and MAE are higher than at the selected checkpoint. This is descriptive evidence supporting checkpoint restoration. Validation itself selects these checkpoints and the blend; it cannot provide an independent estimate of overfitting or new-data generalization. Historical-test interpretation belongs to the parent agent's one frozen retrospective evaluation; no additional tuning or search was performed here.
