# Original dataset enrichment and quality check

The user explicitly chose enriching/checking the existing dataset to retain the endterm's same-dataset requirement. Produce an auditable derived dataset, preserve the original source CSV, all 2,553 retained ad IDs/prices and the seed-42 split, and add no synthetic rows or external specifications. Never repair suspicious values by guessing or remove difficult evaluation rows.

- [ ] Audit TRAIN-only source patterns and quality issues independently. Freeze conservative row-local rules before applying them across the existing fixed split.
- [ ] Add only justified literal fields and clearly labelled mathematical proxies, with unknown/conflict states and source limitations. Keep raw fields and actual reported mileage/prices. Export a data dictionary, provenance/hashes, fixed split metadata and review queue. Target/identity/audit metadata must be explicitly excluded from candidate predictors.
- [ ] Add behavioral tests first for extraction boundaries, unknowns, absence versus negative claims, row/target/split preservation, price/identity independence and export round-trip. Run the existing full suite.
- [ ] Use a bounded TRAIN-only nested three-fold comparison to check candidate feature groups if appropriate; all preprocessing/checkpoints fit inside the relevant folds. No validation/test targets inform this enrichment round and no model promotion based on these reused test results. Do not claim accuracy gains merely from more columns.
- [ ] Integrate deterministic export into the reproducible project workflow, document limits and the model status, independently review, and publish the derived dataset plus complete project to the authorized folder with backup/concurrency checks.

The existing selected model and its 15 comparison rows are a frozen reference. Newly proposed fields are candidate representations, not verified vehicle facts. Any unchanged model metrics must remain explicitly unchanged. Raw source descriptions are capped in most TRAIN rows; missing text cannot be recovered locally.

Ruling: original user authorization and chosen mode cover this work; no separate permission step is needed. Use an isolated artifact directory rather than a git worktree because this project has no git checkout. The dataset improvement itself does not justify selecting another model on the known historical test.

Ruling after TRAIN utility study: only age/usage improved both metrics in all three outer folds. Advanced only that pair to a separate bounded 15-candidate validation follow-up, then froze one winner before a single historical evaluation. Both observed requested metrics improved slightly; search stopped. Integrating the frozen model alongside the enriched export. This is an adaptive extension beyond the descriptive utility study, not preregistered model promotion inside that study. Original rows/prices/splits remain unchanged.
