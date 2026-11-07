# TRAIN-only literal source-field audit

Scope: the current `data.load_data` and `data.split_data`, selecting only the 1,787 original seed-42 TRAIN rows. Inspected predictors were `desc`, `brand`, `model`, with `ad_id` used only for evidence references. The source loader necessarily executes the unchanged deterministic cleaning, including its valid-target filter, but this audit did not inspect price values or analyze targets. No validation/test rows, result/prediction tables, model fitting, new listings, network sources or product edits were used. The unchanged CSV has SHA-256 `b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468`; clean row count remains 2,553. Sorted TRAIN listing-ID fingerprint is recorded in `evidence.json`.

`proposal.py` contains pure per-row functions. `run_audit.py` reproduces the evidence and executes 15 source-field and nine lexical synthetic semantic fixtures. All passed. Fixtures are labeled synthetic: they are safeguards for future input, not a claim that every wording type occurred in TRAIN. No accuracy improvement has been measured or promised.

## Exterior source label: justified with a narrow position rule

There are **1,281/1,787 rows (71.68%)** with one complete recognized label in the exterior slot: immediately after an exact comma-delimited `КПП автомат/механика/вариатор/робот` clause and optional mileage. Both source mileage forms must be skipped: `с пробегом 216 000 км` and bare `81 362 км`. Bare mileage occurs in 67 TRAIN descriptions; 66 of those have a recognized exterior label. Skipping only `с пробегом ...` loses those 66 rows.

The previous 16 conventional labels reproduce exactly **1,265 rows**. An additional 16 rows have exact source labels `вишня` (6), `хамелеон` (6), `бронза` (4). These are defensible as **literal source labels** and should retain their original wording, without converting them into inferred standardized hues or paint codes. Proposed name: `exterior_color_literal`. If a narrower 16-label field is preferred, keep the other three as unrecognized/not-stated rather than guessing their normalized hue.

Counts: белый 407; черный 295; серебристый 176; серый 169; синий 71; зеленый 42; красный 19; бордовый 18; золотистый 17; бежевый 13; голубой 12; коричневый 11; вишня 6; жёлтый 6; хамелеон 6; бронза 4; бирюзовый 4; фиолетовый 3; оранжевый 2. **506 are `not_stated`; zero observed conflicts.** Every recognized exact color clause anywhere in TRAIN also meets this positional rule; that supports the observed source format, not a precision estimate on unseen data.

Do not scan later prose for colors. Real TRAIN ad `206703737` says `КПП автомат, кожа, кондиционер, Европеец кожа салон черный...`: proposed exterior result is `not_stated`, with positive leather/air-conditioning source clauses. The black reference is interior narrative. Another TRAIN description starts the post-КПП section `Голубой металлик`; it remains unrecognized because it is not an exact isolated source label. This loses a plausible literal hint intentionally rather than expanding into narrative parsing.

Conflicting exact recognized colors within the same contiguous recognized source block produce `conflict`, not an arbitrary first/last choice. No such conflict was observed in TRAIN. A negated clause (`не черный`), incomplete token (`черн...`), missing/ambiguous КПП boundary or unsupported layout produces `not_stated`. These latter behaviors are covered by synthetic examples.

The closed vocabulary reflects only the observed TRAIN labels. An unseen otherwise-valid spelling/color is conservatively unrecognized; this is not an exhaustive color ontology.

## Equipment: source clauses justify a small, explicit subset

The prototype reads a **contiguous block of complete exact source option clauses**, starting at the same post-КПП/post-mileage position and optional exterior label. It stops at the first unknown, negated, truncated or narrative clause. A closed 55-clause vocabulary permits continuation through source options (including the literal paint-finish label `металлик`); it is not a proposal to add 55 predictors. No generic word scan, fuzzy synonym expansion, condition/tax claims, description embedding or price-bearing narrative is used.

**1,237 TRAIN rows** contain at least one recognized source option. The clearest bounded candidates are:

| Literal clause | Strict block count | Any complete exact comma clause anywhere |
|---|---:|---:|
| литые диски | 642 | 642 |
| люк | 194 | 194 |
| велюр | 170 | 170 |
| кожа | 139 | 140 |

If implemented, names such as `alloy_wheels_clause`, `sunroof_clause`, `velour_clause`, `leather_clause` should mean **positive exact source wording only**; zero means no supported positive clause, not verified absence. Existing `leather_claim` in the full feature view is broader; the structured version is narrower and partly redundant. A bounded implementation can expose these source columns without asserting they improve prediction. `панорамная крыша` is another explicit clause (36 rows), but a broad panorama feature already exists in the full view. Remaining literal counts are in `evidence.json` for traceability, not an invitation to add an uncontrolled feature set.

The one excluded complete `кожа` clause is ad `204480554`: `..., КПП автомат, с пробегом 33 000 км, 1, 6 акпп, кожа, датчик слепых зон, автопило...`. Parsing stops at `1`; the malformed/narrative sequence cannot safely be treated as a continuous source equipment list. Ad `206781256` has `..., КПП механика, 2.8 турбо дизель 6 ступ коробка механика, кондиционер, ...`; its exact later air-conditioning word is similarly not accepted. Ad `206470561` has narrative `Радной краски, Инжектор, гур...`; it does not become a structured steering-assistance claim. The parser does recognize isolated literal `турбонаддув` as a block-continuation clause, without inferring a turbo specification from model names or generic prose.

Negated clauses (`без люка`, `люка нет`) terminate the block and yield no positive sunroof claim; these are synthetic safeguards. A real TRAIN narrative includes `Отс. Комплектация полная без панорамы ЛЮКС...`, which yields no structured panorama/sunroof clause. Description truncation is common: `литые дис...` must not become a positive exact clause. Omitted later equipment is unknown, and exact positive seller wording remains unverified externally.

If choosing one literal upholstery category rather than separate positive flags, do not select one material arbitrarily. Restrict material labels to `кожа`, `велюр`, `алькантара`, `комбинированный`; `дерево` is not assigned an upholstery material. There are 1,466 omissions, 158 single велюр, 130 single кожа, 14 single комбинированный, four single алькантара, and **15 rows with multiple material labels**. Multiple labels may describe authentic mixed materials, so `multiple_literals` is more accurate than asserting a real-world contradiction. If the implementation uses `conflict`, document it as ambiguity for a single categorical representation, not verified false source wording. Full material-set counts and one observed example are in `evidence.json`.

## Model pooling: lexical prefix labels only, retain original identity

There are 436 raw model names; 210 are singletons; 348 categories with fewer than five rows cover 571 TRAIN rows. The proposed extra `brand_model_family` category pools syntax while retaining raw `brand`, `model`, and `brand_model` columns:

* BMW: only complete three-digit names `^([0-9])[0-9]{2}$` become `BMW | numeric_prefix_<digit>`. This matches 65/110 BMW rows (19 distinct models). `numeric_prefix_5` has 47 rows; prefix 3 has 10, prefix 7 has 4, prefixes 4 and 6 have 2 each. `X5`, `X5 M`, `M5`, `i7`, etc. retain distinct original identities. No numeric prefix is renamed to a catalog series.
* Mercedes-Benz: `^([A-Za-z]+)\s+[0-9]` becomes a literal alphabetic-prefix label. It matches 116/127 rows (42 models): literal E prefix 54, S 27, C 14, G 9, etc. `GLE Coupe 53 AMG`, `AMG GT`, `Sprinter`, `Vito`, `190`, `SLS AMG` remain their original distinct names.
* Lexus: the same syntactic prefix rule matches 73/73 rows (19 models), with literal RX 28, ES 16, LX 11, GS 8, GX 4, LS 3, NX 2, IS 1. Suffixes such as `F Sport Design` remain in the original model column; the coarse field deliberately retains only the initial literal prefix.

Total matched rows are **254**, not all 310 rows of those three brands. On all TRAIN rows, distinct brand/model identities are 441; the extra brand-scoped lexical field has 384 categories. Other brands and unmatched names use `brand | original_model:<original model>`, so Land Cruiser and Land Cruiser Prado are not merged. Namespace markers prevent a raw model string from colliding with a generated prefix category. There is no catalog lookup, learned equivalence dictionary, engine/generation inference or target statistic. Prefix sharing is a transparent representation, not a claim that pooled models are mechanically identical.

## Integration limits

The justified final scope is a positional literal exterior field, a small explicitly named structured equipment subset, and an additional lexical family label that preserves the original model columns. Keep learned category grouping/imputation/scaling within existing training pipelines. Keep original CSV bytes, accepted rows/IDs, target, seed and split unchanged. Unknowns and conflicts must remain visible rather than becoming negative equipment facts or guessed colors. This audit establishes extractability and source semantics only; it does not establish predictive value, independent generalization or a gain in either MAE or within-20% accuracy.
