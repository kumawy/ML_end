# Focused copy of the original course dataset

`used_cars_enriched.csv` preserves all 32 loaded source columns, every retained ad, original asking price, row order and historical seed-42 split. It adds literal steering, three fixed-2025 usage proxy fields and `source_row_index`/`split` metadata, for 38 columns. The original CSV is unchanged.

`vehicle_age_proxy` is 2025 minus a valid positive integral model year; it is audit only. `log_vehicle_age` is log1p(age), and `log_annual_mileage_proxy` is log1p(reported km / max(age, 1)). Invalid or unavailable years and unavailable/invalid mileage remain unknown in the relevant proxies. These are mathematical proxies, not measured time since registration or annual usage. Steering comes only from exact pre-КПП source clauses; omitted steering remains unknown.

`data_dictionary.csv` describes every field and marks the union of selected-branch fit inputs. Targets, IDs, URLs, raw free text and split metadata are excluded. Original auxiliary fields remain for audit. `quality_report.json` records provenance hashes, missing counts, split counts and proxy semantics. This export fits no transforms or models; model preprocessing is fitted inside training folds. Model metrics and the historical-evaluation caveat are documented in the project README and `results/report.json`.
