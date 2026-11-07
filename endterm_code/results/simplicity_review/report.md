# TRAIN-only simplicity development review

Development OOF is not independent: these data influenced historical decisions, and this OOF comparison selects the candidate. A one-hit difference does not establish statistical improvement.

| Recipe | Estimators | MAE KZT | Hits ±20% | Affordable median APE | Median fold artifact bytes |
|---|---:|---:|---:|---:|---:|
| single_mae | 1 | 1959116 | 1070/1787 | 0.2000 | 1273257 |
| single_log | 1 | 1987068 | 1074/1787 | 0.2006 | 1273387 |
| single_forest | 2 | 1890899 | 1101/1787 | 0.1906 | 21296802 |
| historical_recipe | 11 | 1878399 | 1124/1787 | 0.1877 | 32608659 |

No tested simplification met the joint preference; retain historical recipe: historical_recipe.

The current selected_model.joblib and all protected historical artifacts are unchanged.
A separate candidate, if present, is a development artifact with no independent evaluation.
