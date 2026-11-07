# Requirement-to-evidence map for both course regulations

Reviewed against the user-provided course PDFs on 7 November 2026. The PDFs are assignment specifications, not instructions to execute external actions. This map describes evidence and remaining limitations; it does not predict a grade.

## Source documents and stages

- `Midterm Course Project.pdf`, SHA-256 `b7586ad50122f08cb8f16a59c795a35553983591ee35a82477861944c9fddaa4`: one course-long project; intermediate defense; historical presentation 7–12 slides / 10 minutes.
- `Endterm Course Project.pdf`, SHA-256 `267490c184fbda619a321a5ebc8c87d74b95abb2809ed18eed088042a6758234`: continue the same dataset/theme/group; final presentation 10–15 slides / 12 minutes, then questions.

The current 15-slide submission follows the endterm format. The earlier midterm slide limit does not reduce the final deck to 12 slides. Neither PDF specifies a submission deadline or an upload location.

## Requirements and current evidence

| Requirement | Evidence and status |
|---|---|
| One course project; unchanged theme | Kazakhstan used-car price estimation retained. Asking price is explicitly identified as a proxy, not a verified sale value. |
| Same real dataset and group | Original Kolesa CSV hash, 2,553 clean IDs/targets, and Alisher Akhmet / Aslan Muratov retained. `data.py`, source hashes and `split_manifest.csv` enforce the data identity. |
| Topic should not repeat another team's | Historical midterm README reported instructor availability confirmation according to the team. Other teams' topics cannot be independently checked from these files. |
| Problem, task, target and success criterion | README defines regression, `price` in KZT, the original MAE-only selection and retained ≥20% improvement versus the training-median baseline. A later user-directed objective requires both lower MAE and higher within-±20% frequency; strict validation eligibility and equal normalized MAE/miss ranking are documented separately. The original criterion is unchanged and remains met. |
| Dataset description, provenance, licence/link and issues | README gives source URL, dimensions, metadata qualifications, hashes and missing information. The 2,553×47 derived export retains the loader's 32 prior columns, IDs, prices, order/split and raw CSV; dictionary, quality report and 598-row inspection queue expose literal claims, unknowns and provenance. No synthetic project observations or external specifications were added; fixtures do not enter the dataset. |
| At least four meaningful EDA plots, each interpreted | Current `explore()` produces `eda_prices.png`, `eda_year.png`, `eda_mileage.png`, `eda_fuel.png`. Interpretations are in README and `report.json` → `eda.plots`; all use training listings only. |
| Documented cleaning and feature engineering | Original exclusions/deduplication/missingness, fuel/engine semantics and strict steering remain. New positional color/equipment labels, narrow lexical families, fixed-2025 age/usage proxies and audit-only flags preserve source values. No equipment mention is NaN/unknown, not verified absence. TRAIN-only source audits and all five utility groups, including worse MAE outcomes, are recorded; no condition, catalogue identity or true mileage is invented. |
| Baseline plus at least two lecture algorithms | Median baseline plus Linear Regression, KNN Regressor and Decision Tree Regressor all remain visible, using unchanged legacy inputs for genuine historical controls. |
| CV for at least one model; justified tuning | KNN/tree/forest use shared training-only three-fold CV; CatBoost has its own explicitly recorded training-only fold protocol. Search candidates and scores are saved. |
| Correct splitting and leakage prevention | Original 1,787/383/383 split, no repeated ad IDs across splits; fitted imputers/scalers/encoders remain train/fold-only. No test labels are used for fitting/checkpointing/candidate selection in this round; the unchanged loader still parses the full CSV for original eligibility. Validation selects configurations/checkpoints/weights. Historical test feedback and the disclosed final retain-baseline acceptance rule make this a retrospective benchmark, not independent confirmation. |
| At least one K-Means or DBSCAN analysis | K-Means k=2..8, scaling, 20 initializations, seed and silhouette rule documented. Profiles/counts/interpretations saved. Displacement geometry uses the justified 1,775 non-electric training subset; all cars remain in prediction. Both clustering algorithms are not required. |
| PCA with interpretation and explained variance | PCA on the same three numeric training features, variance/loadings/projections saved; first two explain 80.13%. It is a visualization, not a replacement for regression. |
| ANN with architecture, loss, optimizer, LR, history and overfitting discussion | Single-ANN controls remain, including `ANN corrected` epoch 71/111. Fixed 307-input steering and 309-input age/usage five-seed 128→64 pools document ReLU/linear output, train-only log-target scaling, Adam/LR/batch/L2 and both checkpoint rules. New selected MAE epochs are 34/56/93/56/31 of runs 79/103/138/103/76; all histories remain. Validation gaps are descriptive, not proof of external robustness. |
| Application-appropriate deep learning | The task is tabular; feedforward ANN is the required deep-learning model. The endterm text expressly does not require forced CNN/NLP/RL for unsuitable data. |
| Preserve required methods when adding models | CatBoost, ANN pools, mixtures and derived fields supplement the lecture models, K-Means, PCA and ANN. All preceding comparison controls remain; optional literal/quality fields do not replace required methods or change accepted observations. |
| Final fair comparison, errors and conclusions | All 16 rows use identical validation/test IDs; all prior 15 are retained. Latest Usage ensemble reduces test MAE 0.70535% and increases within ±20% by 0.78329 pp versus prior Robust; RMSE/R² improve, median absolute error worsens and ±10/30% each lose one hit. Both bootstrap intervals include zero. The earlier 51-candidate search, all five TRAIN utility outcomes including negatives, and separate 15-candidate usage ledger remain visible. These stages add adaptive choices, not independent confirmation. |
| Reproducibility and clear execution order | `run_project.py` remains the ordered `.py` entry point and also exports the enriched dataset; standalone `dataset_export.py` reproduces export without training. Seeds/versions/lock/hashes/histories and 58 passing tests are recorded. The 122.9s current run reproduces the frozen 16-row result. README derives steering and dataset fields before saved-contract inference. Historical 35/54-test counts and earlier clean-install evidence are clearly scoped. |
| Open problems and future plan | Methodology/experiment log/slide 14 disclose sparse models, missing condition/location/dates, historical evaluation and need for independent later-data validation. |
| Three submission artifacts | Runnable `.py` project, `.pptx` and `.pdf` presentation, and `.md` README are included. Both notebook formats are not required. |
| Individual technical contribution and participation | **Not verified.** Historical roles are preserved; new student contributions must be performed and recorded in `CONTRIBUTIONS.md`. AI preparation is disclosed. Both students must speak and answer during the defense. |

## Data-collection boundary

There is no requirement to replace the dataset with a 2026 snapshot or collect a minimum number of new rows. Additional real dated data may support a separately documented external validation, while preserving the original course dataset and required analyses. Do not attach current prices/specifications to an old ad as though they were one historical observation, invent missing fields, or silently reshuffle the original split.

The current loader intentionally validates the original snapshot's year range through 2025. A future 2026 external set would need a separate date-aware preparation/inference protocol, source/provenance and overlap checks; it is not part of the present submission or a promised accuracy gain.

## Items that a generated package cannot complete

Actual personal technical work, instructor topic registration and defense performance cannot be certified by code or this checklist. The midterm and endterm documents both impose a 30-point deduction for missing defense. Keep real contribution evidence and prepare both members to explain the methods and limitations.
