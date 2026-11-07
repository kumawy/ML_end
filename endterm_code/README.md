# Used-car asking-price estimation in Kazakhstan

**Machine Learning Algorithms — endterm.** Team: Alisher Akhmet and Aslan Muratov.

Estimate the advertised asking price of a used car from its vehicle specifications. This is regression, with `price` in KZT as the target. The midterm success criterion is retained: MAE at least 20% below predicting the training median. MAE measures money error; RMSE, R², median absolute percentage error (median APE) and fractions within ±10/20/30% show other aspects of performance. R² is not the proportion of accurate predictions.

## Setup and execution

Use **Python 3.12**. From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_project.py
python -m unittest discover -s tests -v
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`. The pinned core dependencies are in `requirements.txt`; `requirements-lock.txt` records the resolved environment. Training uses the bundled CSV and downloads no data. Paths resolve from the script directory. Each run replaces the current generated results; run a separate checkout if preserving another execution is necessary.

`run_project.py` executes the full workflow in order: data validation and fixed splitting, derived dataset export, four interpreted TRAIN EDA plots, K-Means/PCA, original supervised controls and training-only CV, ANN training, the fixed ensemble, validation evidence, and retrospective test evaluation. To regenerate only the derived dataset:

```bash
python dataset_export.py
```

## Data and preparation

Source: [Kolesa-Cars-2025 on Kaggle](https://www.kaggle.com/datasets/mutashevdarkhan/kolesa-cars-2025). The project continues [AlisherAt/MLMid](https://github.com/AlisherAt/MLMid), snapshot `059b9319ea07b40cd49e5946f4a0b31a9d4e5bce`. The original README records MIT catalog metadata. That metadata is not proof of ownership of third-party listing text or of each ad's observation date.

The unchanged source has **3,999 rows and 14 columns**. Remove 527 new cars, 51 used cars offered on order and 868 repeated remaining ad IDs, leaving **2,553 used-car advertisements**. No synthetic project observations, new target values or difficult-row exclusions were added. The loader checks source integrity:

- SHA-256: `b348551cd876bde7d25fce79c3fcdbd05920403c567a6aedc0e0a5e19e260468`.
- Original Git blob SHA-1: `01e57295927f91ffd7996745ce6a867b6d41b0f7`.

The source represents **2025 seller asking prices**, not verified sales or current 2026 prices. Individual listing dates, structured location, verified mechanical condition and VINs are unavailable. Descriptions are often truncated. There are 497 missing mileages; unreliable original `drive` values are excluded from prediction.

`data.py` keeps literal source semantics: structured fuel wording recovers 16 electric and 109 gas/petrol listings. Missing fuel stays `unknown`. Electric displacement is inapplicable (0 L plus a flag) rather than an ordinary missing combustion-engine value. Positive source conflicts remain visible. Equipment/power fields use supported seller wording, not inferred catalog facts; a missing claim does not prove equipment absence. `structured_features.py` reads only exact steering clauses preceding `КПП`; omitted steering remains unknown.

`dataset_features.py` adds only the selected fixed-snapshot proxies: age = 2025 − valid model year, log1p(age), and log1p(reported mileage / max(age,1)). Missing or invalid source inputs remain unknown. These are mathematical proxies, not measured annual use. The derived export preserves all 32 loaded columns, adds steering, the three age/usage fields and source-row/split metadata: **2,553 × 38**. `dataset/data_dictionary.csv` distinguishes predictors, targets, source text and audit metadata. The selected branches use a union of 25 raw input fields; 38 exported columns do not mean 38 model inputs.

## Evaluation design

The original seed-42 split is unchanged: **1,787 TRAIN / 383 validation / 383 historical test**. No ad ID occurs in more than one split. Relisted vehicles with different IDs cannot be excluded without vehicle identifiers.

Median imputers, scalers, category encoders and ANN log-target scaling fit training rows only. CV pipelines refit preprocessing inside each fold. Training-only 3-fold CV tunes KNN, Decision Tree and Random Forest by negative KZT MAE, including OHE minimum frequency. Validation selects ANN architectures/checkpoints. Price, raw price text, IDs, links and split labels are excluded from model inputs. Clustering and PCA fit only TRAIN, with price used afterwards for descriptive profiles.

**The test is a retrospective benchmark.** Earlier test errors influenced feature choices and previous acceptance decisions, and validation has been reused. This cleanup preserves the frozen chosen recipe and performs no new blend search or test-based acceptance. A correct train-only fit boundary does not erase earlier research feedback. Independent later observations are still needed to establish generalization. Do not present these scores as guaranteed performance on new advertisements.

## Models and analyses

The nine-row comparison retains the original median, Linear Regression, KNN (15 neighbors, distance weights) and Decision Tree (depth 10, minimum leaf size 3) on unchanged midterm inputs. Tuned KNN, Tree and Random Forest use corrected/full predictors and the shared training-CV protocol. `legacy_data.py` exists to preserve genuine original controls.

The single `ANN corrected` uses corrected numeric/categorical vehicle inputs and OHE minimum frequency 2. It compares fixed 64→32 and 128→64 hidden layers. ReLU hidden activations, a linear output, Adam, learning rate .001, L2 alpha .001 and batch size 64 train standardized log1p(price). `ANNRegressor` inverts predictions into nonnegative KZT. At most400 epochs and validation-MAE patience 40 select the checkpoint. `ann_history.csv/.png` retain training/validation loss and MAE. The report discusses convergence and a descriptive post-checkpoint overfitting heuristic.

The retained `ANN ensemble` uses five corrected-plus-steering networks and five age/usage networks, each with 128→64 hidden layers, fixed seeds 11/23/42/71/101 and cap 300/patience 45. Each trajectory retains both KZT-MAE and normalized MAE/miss checkpoints; both criteria affect stopping. The fixed prediction formula is:

```text
0.25 × usage MAE pool + 0.75 × (0.75 × corrected dual pool + 0.25 × Random Forest)
```

Each usage network has weight .05, each corrected dual network .1125, and RF .1875. The normalized dual score averages MAE/reference-MAE and miss-fraction/reference-miss-fraction. The corrected checkpoint reference is .75 × single corrected ANN + .25 × RF; the usage checkpoint reference is .75 × corrected dual pool + .25 × RF. These training references are not additional reported contenders. Histories and exact parameters remain in `results/robust_ann/` and `results/usage_ann/`. Only the final model bundle is saved; intermediate fitted copies are unnecessary.

K-Means uses TRAIN year, log mileage and displacement with median imputation and StandardScaler. The 12 electric TRAIN listings remain in supervised modelling but are excluded from displacement geometry, leaving 1,775 clustering rows. Compare k=2..8, n_init=20, seed 42 and select by silhouette. Profiles interpret the groups without claiming universal market segments. PCA uses the same scaled features; loadings, projections and explained variance are retained. Neither method replaces the price predictor.

## Results and limitations

The complete comparison is `results/model_comparison.csv`; predictions for identical validation/test IDs are in `all_validation_predictions.csv` and `all_test_predictions.csv`. The consolidated report records parameters, scores, EDA interpretations and environment. Selected historical-test results:

| Metric | Result |
|---|---:|
| MAE | 1,634,564 KZT |
| RMSE | 3,189,732 KZT |
| R² | 0.9165 |
| Median APE | 16.61% |
| Within ±20% | 224/383 = 58.49% |

MAE is 72.53% below the training-median baseline, meeting the retained project criterion. Error files show price groups, missing mileage, fuel, model support and largest errors. Price-band cutoffs 3.6/6.7/11.5m come from TRAIN and serve only evaluation. Premium cars, unseen models and rare groups have limited support; a high R² can coexist with many individual errors. Asking prices, missing condition/location/dates and possible relisting dependence limit practical use. Use estimates as historical reference information rather than as transaction values or automated purchase decisions.

Future work is independent dated data, confirmed source corrections and improved coverage of sparse vehicle groups. The earlier experiments, including negative results and the unsuccessful simplicity review, remain in [Git history at 6abb0ea](https://github.com/kumawy/ML_end/tree/6abb0ea). Removed files do not erase those outcomes or establish that the current recipe was chosen without historical feedback.

## Saved-model use

Run from this directory in the same dependency environment:

```python
import joblib
from pathlib import Path
from data import load_data, split_data
from dataset_features import add_dataset_features
from structured_features import add_structured_features

root = Path('.')
frame, _ = load_data(root)
bundle = joblib.load(root / 'results/selected_model.joblib')
query = frame.loc[split_data(frame)['test'][:3]].copy()
query = add_structured_features(add_dataset_features(query))
inputs = {'enhanced': query[bundle['features']['enhanced']]}
prices_kzt = bundle['estimator'].predict(inputs)
```

This example reproduces saved historical predictions; it is not independent evaluation. New inputs need the same engineered column semantics. Reuse fitted transformers without refitting at inference.

## Team responsibilities

Responsibilities follow the midterm split:

| Member | Assigned technical scope | Defense sections |
|---|---|---|
| **Alisher Akhmet** | Dataset provenance, cleaning, feature semantics, EDA, derived export, K-Means and PCA | Data, preparation, exploratory analysis and unsupervised methods |
| **Aslan Muratov** | Baselines, training CV and tuning, ANN and ensemble, metrics, error analysis and reproducible execution | Supervised methods, neural networks, evaluation and conclusions |

This table allocates responsibility; it does not invent completed personal runs, commits or dates. AI assisted code, analysis, documentation and slide preparation. Each member must personally review and reproduce their assigned sections, explain the methods and limitations, and speak during the defense. Actual technical contribution and participation are course requirements that generated files cannot establish.

The final deck has 14 slides and a 670-second speaking plan within the 12-minute limit. See `DEFENSE_NOTES_RU.md` and `REGULATION_COMPLIANCE.md`. Submit the runnable `.py` project, presentation and this README as required by the instructor.
