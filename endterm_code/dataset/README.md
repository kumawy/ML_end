# Enriched copy of the original course dataset

`used_cars_enriched.csv` retains every eligible source ad, original asking price and original split. The original CSV remains unchanged. `source_row_index` and `split` are audit metadata; prices, IDs, URLs and raw free text are excluded from candidate predictors. See `data_dictionary.csv` for every field and its missing-value semantics.

The extra fields are candidate representations and review signals, not recovered ground truth. Unknown equipment remains missing; a numeric model must retain training-fitted missingness indicators. Literal upholstery/color labels and lexical families do not certify condition, catalog equivalence or specifications. Mileage/age is a fixed-2025 mathematical proxy, not measured annual usage. Suspicious mileage/drive values are retained in `review_queue.csv` for human checking; none are repaired or removed.

`quality_report.json` records row/split counts, source/export hashes and field coverage. Source-pattern discovery used TRAIN only; the fixed row-local rules are then exported for all original splits. There are no fitted transformations in this export. Actual field usage and forecast metrics are described in the project README and results/report.json. Extra columns alone are not evidence of more accurate forecasts; the selected model need not consume every exported candidate.
