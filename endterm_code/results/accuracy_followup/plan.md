# Dual-metric accuracy improvement plan

Goal: reduce original-KZT MAE and increase the fraction within ±20% simultaneously, as explicitly chosen by the user.

Constraints: same original CSV, row eligibility, ad IDs, seed-42 1787/383/383 split, theme, group and required lecture models/KMeans/PCA/ANN. No synthetic project rows, external specifications, hard-row deletion or price-derived input features. Historical test has already influenced development and remains retrospective.

1. Read historical experiments and audit unused original structured fields using TRAIN only. Independent audit, bounded CatBoost CV and bounded alternative-model CV run in separate scratch folders.
2. Compare a fixed ANN variance-reduction experiment: legacy/full/corrected representations, five fixed seeds (11,23,42,71,101), one 128→64 architecture, MAE and dual-score checkpoints; average all five members for each of six candidates. All fitted transformations use train only. Validation controls checkpoints. This is a scratch experiment, not yet product code.
3. Before test access, freeze a small shortlist and explicit selection rule. Candidate eligibility requires validation MAE below baseline and validation within±20% above baseline; rank eligible candidates by equally weighted normalized MAE and miss fraction. Keep baseline if none eligible. Any added blend grid must be recorded before evaluating it. Record every result, including failures.
4. Only after selection is frozen, evaluate the one selected final candidate on the known historical test once, with paired bootstrap uncertainty as descriptive evidence. No further test-guided search. A candidate must improve both observed metrics for the requested positive outcome; otherwise retain the delivered baseline and document limitations.
5. If successful, integrate reproducible training/prediction with meaningful tests for the new boundary, run full project and fresh tests, review independently, update documentation and presentation consistently, then publish to the already authorized user folder with backup and concurrency checks.

Baseline: validation MAE 2,289,371.696786993 KZT and within±20% 0.6422976501305483; historical-test MAE 1,667,212.1367344395 and within±20% 0.5613577023498695. Success on one split does not establish new-data generalization.

Ruling: user already authorized implementing accuracy improvements and requested autonomous work. No separate planning approval needed. Existing project is a local artifact directory, not a git repository; isolated private copies replace git worktree operations. Low-impact scratch experiments require result checks rather than mirrored unit tests. Product integration will add meaningful behavioral tests.

Progress: reconnaissance and parallel experiments in progress.
