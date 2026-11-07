# Enriched-dataset presentation completion

Frozen outputs, 2026-11-07:

- `outputs/ML_Endterm_Enriched_Dataset.pptx`: 278,361 bytes. SHA 256 `28f41a64ff6428e727bc07ab88505da977c718db1e69d8312a2710afaf43c596`.
- `outputs/ML_Endterm_Enriched_Dataset.pdf`: 3,682,910 bytes. SHA 256 `bf548b8ea811404e7b4ee5b4a02a968e02e8f8573123215ed71d9b1dc5b22e6b`.

The final PPTX matches `slides/final-v2.validation.json` and `slides/revision2/final.pptx`. It contains exactly 15 slides,8 native editable charts and 5 native tables. The finalizer reports 0 layout findings,0 warnings, successful native-chart validation with embedded literal-data workbooks and successful first-party import. Arial and 16:9 geometry match the reference. No PowerPoint UI validation was claimed.

Every exact final PPTX slide was rendered and individually inspected at 1920×1080. Slides 1,2,4,5,7,8,9,15 are pixel-identical to the prior reference. Slides 3,6,10–14 carry the requested enrichment, selection, architecture/history, sixteen-model comparison and mixed-gain narrative. The full native table contains 16 models and 7 columns, with 96 displayed numeric metrics. Sources verify all 96 values and all 8 native charts against the frozen stage CSV/JSON/history. See `slides/source_value_verification.json` and executable checks.

The 15 speaker-note narratives total 700 nominal seconds (11 m 40 s), with source citations. Notes 1,3,15 cite both supplied midterm and endterm regulations. The 58-test count comes from the parent’s completed current-stage verification. Historical documented roles and AI assistance remain explicit. New personal student contributions remain unverified.

Current selection is 25%new 309-input five-seed MAE pool plus 75%prior Robust ensemble, equivalently five new ANN weights 0.05 each, five prior 307-input ANN weights 0.1125 each and forest 0.1875. Slide 11 uses the new 309-input seed 42 MAE checkpoint 93/138 and its actual training/validation curves. The original ANN controls, prior pools, CatBoost and negative alternative experiments remain in the evidence.

The benchmark conclusion stays narrow: MAE 1,646,174.8055 to 1,634,563.5634 KZT (−0.7053%), within±20%221 to 224 of 383 (+0.7833 pp), improved RMSE/R², slightly worse median absolute error and one fewer hit within±10%and±30%. Both paired 95%gain intervals crosszero. Notes explain the known historical split, repeated validation and excluded adaptive-selection uncertainty. No independent generalization claim or further search is implied.

The direct Artifact Tool PDF backend still lacks `#standard-fonts/Helvetica`. The PDF therefore embeds lossless copies of the inspected exact final-slide PNGs. All 15 embedded images are pixel-identical to those PPTX renders, with 15 verified 960×540 pt pages. Poppler also rendered all 15 pages successfully. The PPTX retains native editable evidence. See `slides/pdf_verification.json`.

Only private presentation sources/assets and the two new outputs were changed. Previous outputs, code/results, docs and the user folder were not edited. One successful edit operation-start marker preceded authoring. The builder guard requires 16 rows and Usage ensemble, preventing stale 15-row data. Source and output hashes are preserved in `slides/freeze_manifest.json`.

Source hashes:

- `work/dataset_enrichment/code/results/report.json`: `bbf87b26c6fbb07e24f08b1b36186ff9d65d4ff2d626ccf62034b2ab68f82584`
- `work/dataset_enrichment/code/results/model_comparison.csv`: `782511a39ddd2cc7b8127d6e390432f6fc06327ec7651ce044cebea9b8a80080`
- `work/dataset_enrichment/code/results/usage_ann/robust_ann_history.csv`: `2f3e56cca8b41c3d210761c290ad04091874d2d4ec12b4d78d9c2dd4f74011f4`
- `work/dataset_enrichment/code/results/split_manifest.csv`: `81b60d1c3aa9d1c56e6d421c32e9668d376c710ff36fdedce871ed57f6644efb`
- `work/dataset_enrichment/code/dataset/quality_report.json`: `bae60a3366bc24983ff1b010c8acff811ad9e3eed8ca06a360ecfab204c17fc9`
- `work/dataset_enrichment/usage_validation/frozen_evaluation.json`: `93c80faef092dcbd2a8caf006477ff42b28a6a56d551e0642d2931f699c03b0c`
