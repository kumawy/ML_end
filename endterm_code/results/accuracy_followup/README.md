# Historical development evidence

These retained development records document the bounded search and the frozen selected model before its one retrospective test evaluation. The original CSV is loaded for unchanged cleaning/splitting; test targets do not determine fitting, checkpoints, candidate ranking or blend weights. A `test_accessed: false` field in older scratch logs means no test evaluation/selection, not that the raw source CSV was never parsed.

The 51 validation candidates comprise 19 new individual candidates, the two existing ensemble/forest controls and 30 fixed blends. These are reused development observations, not independent evidence of generalization. Paired bootstrap intervals include zero and ignore selection bias.

Historical JSON artifact paths are provenance, not promised packaged models. Unselected exploratory model binaries are omitted. The runnable project entry point reproduces all 12 prior comparison controls, both fixed steering ANN pools and the final six-mixture follow-up. The full search grids, CV metrics, validation vectors and frozen decision remain here for audit. Final training and prediction have no dependency on these development artifact paths.
