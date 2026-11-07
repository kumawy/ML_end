# Course requirements and submission evidence

This map uses the supplied Midterm and Endterm Course Project PDFs. The endterm continues the same project, real dataset and two-person team. The documents define assignment requirements; they do not certify a grade or personal participation.

## Source specifications

- Midterm PDF SHA-256: `b7586ad50122f08cb8f16a59c795a35553983591ee35a82477861944c9fddaa4`.
- Endterm PDF SHA-256: `267490c184fbda619a321a5ebc8c87d74b95abb2809ed18eed088042a6758234`.

## Requirement mapping

| Requirement | Current evidence |
|---|---|
| Same theme, real-world dataset and team | Kazakhstan used-car price regression, original Kolesa CSV, Alisher Akhmet and Aslan Muratov. Source hash, 2,553 cleaned IDs/targets and historical split are preserved. No synthetic project observations. |
| Clear problem, target and success metric | README: advertised price in KZT; MAE plus RMSE/R²/median APE/tolerance rates; retained ≥20% MAE reduction against the training-median baseline. |
| Dataset source, size, known issues and acquisition | README and loader identify Kaggle/original midterm, source integrity, loading/filtering and limitations. The focused 38-column export preserves the 32 loaded columns and all retained rows. A dictionary and quality report separate predictors from target/metadata. |
| At least four meaningful EDA plots with interpretation | TRAIN price distribution, year/median price, mileage/price and fuel/price distributions. PNGs/underlying CSVs in results; `report.json` → `eda.plots` includes interpretations. |
| Cleaning and justified feature engineering | Exclusions/deduplication, missingness, literal fuel/engine/steering, log mileage, encoding/scaling and fixed-2025 age/usage proxies are documented. Unknowns remain unknown. Source labels/prices are not replaced or removed to improve scores. |
| Train/validation/test design and leakage-safe preprocessing | Original 1,787/383/383 split and unique ad IDs; imputation/scaling/encoding fit TRAIN only, inside CV folds where applicable. ANN target scaler is TRAIN-only; validation controls checkpoints. Historical test reuse is explicitly disclosed below. |
| Midterm baselines and lecture algorithms | Unchanged median, Linear Regression, KNN and Decision Tree controls remain through `legacy_data.py`; no lecture method was replaced by optional extensions. |
| Cross-validation and tuning | KNN, Tree and Random Forest use training-only 3-fold CV. Parameter grids, best settings and CV scores are saved. |
| K-Means or DBSCAN, justified features/scaling/parameters and interpretation | K-Means on year/log mileage/displacement, TRAIN median imputation and scaling, k=2..8, n_init=20, seed=42, silhouette selection, profiles and counts. Non-electric displacement geometry is justified; electric rows remain in prediction. Both clustering algorithms are not required. |
| PCA, visualization and explained variance | Same TRAIN numeric view; saved projections, loadings, variance chart and interpretation. PCA supports exploration rather than replacing supervised prediction. |
| ANN architecture, training and fair comparison | `ann.py`: fixed 64→32/128→64 alternatives, ReLU/linear output, standardized log1p target, Adam/.001 LR, batch size 64/L2=.001, train/validation histories, original-KZT checkpoint selection and overfitting discussion. Same validation/test IDs as baselines. |
| Appropriate deep-learning component | The regulation explicitly accepts a feedforward ANN for ordinary tabular data. CNN/NLP/RL need not be forced onto unsuitable data. The fixed ANN ensemble supplements the required single-ANN control. |
| Final comparison, errors, conclusions and practical considerations | Nine required/used models in `model_comparison.csv`; aligned per-ID predictions, price/support/missingness/other error groups and largest errors. README/deck discuss asking-price semantics, missing information, sparse groups and historical evaluation. |
| Reproducible ordered code | `run_project.py` runs the complete workflow using local CSV, fixed seeds, pinned versions and recorded environment. Focused tests protect actual retained behavior. `dataset_export.py` reproduces the derived table independently. |
| Required submission formats | Ordered `.py` project, `.md` README and `.pptx`/`.pdf` presentation. Endterm requires 10–15 slides and 12 minutes, followed by questions; the midterm's earlier 7–12 limit does not govern the final deck. |
| Team roles and identifiable technical contribution | README allocates Alisher's data/EDA/K-Means/PCA scope and Aslan's models/CV/ANN/evaluation scope, consistent with midterm. The allocation does not fabricate personal completion evidence. Each student must perform and explain identifiable technical work. |
| Participation and topic approval | Both members must speak and answer during the defense. No defense incurs −30 points. Actual participation and uniqueness/approval of the topic across other groups require human confirmation. |

## Evaluation limitation

Earlier test errors informed development and validation has been reused. The test metrics are a retrospective benchmark, not independent new-data confirmation. This cleaned implementation fits no model or preprocessing on test, freezes the retained ensemble before test evaluation and does not accept/reject it based on test scores. These boundaries preserve reproducibility while honestly disclosing the earlier feedback; they cannot restore test independence. Independent dated data remains future work.

A new 2026 dataset or 90% predictive accuracy is not required by either PDF. Extra real observations may support later external validation, but must not silently replace the original course dataset or attach current facts to old advertisements. No generated file can certify a student's personal contribution, defense performance or the instructor's evaluation.

Earlier optional experiments and negative outcomes are retained in Git history at commit `6abb0ea`; they are removed from the working submission to keep its execution path and presentation clear.
