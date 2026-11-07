# Frozen accuracy-round presentation

Final files:
- /Users/muratovaslan/Documents/Codex/2026-11-07/z/outputs/ML_Endterm_Accuracy_Improved.pptx
- /Users/muratovaslan/Documents/Codex/2026-11-07/z/outputs/ML_Endterm_Accuracy_Improved.pdf

The deck retains 15 English slides, the original green/offwhite Arial design, both provided regulation citations and a 700-second nominal defense plan. It includes approximately 1,290 narrative words, honest documented historical roles, AI assistance and 35 regression tests.

The new final model is explicitly 75% of the five-seed corrected-plus-literal-steering dual-checkpoint ANN pool and 25% of the prior random forest. The pool uses all five seeds 11/23/42/71/101, 307 inputs, 128/64 ReLU layers and a linear output. Slide 11 clearly labels seed 42 as a representative learning curve, saved at dual-score epoch 69 of 117. Notes record all five checkpoints and run counts, with the prior single-ANN history retained in project results.

Slide 12 includes all 15 models and 90 displayed values: validation MAE and within ±20%, test MAE and within ±20%, test RMSE and R². Validation RMSE and R² remain in the full results CSV, as the slide states. Required legacy baselines, K-Means, PCA, ANN controls, the MAE-checkpoint pool, CatBoost and the archived unsuccessful alternative comparisons remain represented. Slide 6 and notes retain the 20 CatBoost / 12 alternative CV screen and 51 final validation-candidate ledger, without pretending that this count includes all epochs or CV fits.

Slide 14 compares against the requested prior delivered Ensemble, rather than emphasizing a weaker historical comparator. It shows MAE 1.667m to 1.646m KZT, 1.26% lower, and 215 to 221 of 383 within ±20%, a 1.57-percentage-point increase. Median APE improves 17.25% to 16.63%. The adverse RMSE 3.17m to 3.25m and R² 0.918 to 0.913 remain visible. Both descriptive paired 95% intervals cross zero, displayed as approximately −73k to +104k KZT for MAE reduction and −1.31 to +4.44 percentage points for within-20 gain. The deck calls the gains small and uncertain and does not claim new-data generalization or significance.

Validation evidence:
- Finalizer receipt: slides/final-v2.validation.json. Package integrity, geometry, fonts, native chart/workbook correspondence, chart titles and first-party re-import passed. Result: 15 slides, 8 native editable charts, 5 native editable tables, zero findings and zero warnings.
- Every exact final PPTX slide was rendered at 1920×1080 and inspected individually. A stale historical-MAE wording on slide 5 was corrected to distinguish the midterm comparison from the new two-goal rule.
- slides/source_value_verification.json records 25 numerical source comparisons covering all 8 chart series sets, all 90 displayed model metrics, frozen before/after values, seeds, checkpoints, steering counts and uncertainty/tradeoff labels. Values match the completed stage report, CSVs and frozen_evaluation.json after declared display rounding. Chart data retain 12 significant digits for Excel-compatible literal workbooks.
- slides/notes_verification.json confirms 15 sourced notes, exactly 700 nominal seconds, both regulations, all four EDA files, 35 tests and the precise current-round test-use caveat. No test predictions or labels were used for fitting/checkpointing/candidate selection in this round; historical test reuse is disclosed.
- PDF has 15 pages at 960×540 pt. All 15 lossless embedded images are pixel-identical to the inspected exact final PPTX renders. Every page was re-rendered with Poppler; the uncertainty page was visually inspected. Evidence: slides/pdf_verification.json.

The bundled direct PDF backend lacks #standard-fonts/Helvetica, so the delivered PDF follows the previously validated lossless slide-image route. Text, charts and tables remain editable in the PPTX; PDF is the visual export. Native PowerPoint/Google Slides opening was not performed.

Frozen SHA-256 hashes:
- PPTX, 275,135 bytes: 852f182091ae4930ca1a9d79e75976776c6e58933e8d46b672001ee080597263
- PDF, 3,614,952 bytes: e887c693d63e5cdc2b1d832f9d613886d1b114d35c1c20dbeb6c619971eaf93b

Only new slide sources/private evidence, these newly named output files and this report were modified. Code, results, documentation and user-folder deliverables were not modified by the presentation task. Artifacts are frozen for parent packaging.
