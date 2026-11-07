# ML_end

Machine Learning Algorithms endterm: **used-car asking-price estimation in Kazakhstan**.

The project continues the original midterm dataset and team. It includes the required supervised baselines, cross-validation and tuning, K-Means, PCA, a feedforward ANN and the retained ANN ensemble.

- [Code, setup, results and team responsibilities](endterm_code/README.md)
- [Presentation](ML_Endterm_Presentation.pptx) · [PDF](ML_Endterm_Presentation.pdf)
- [Course requirement mapping](endterm_code/REGULATION_COMPLIANCE.md)
- [Russian defense notes](endterm_code/DEFENSE_NOTES_RU.md)

The source is a 2025 advertised-price snapshot. Historical test feedback informed earlier development, so the recorded results are retrospective and do not establish accuracy on new market data.

From `endterm_code`, install `requirements.txt` in Python 3.12, run `python run_project.py`, then `python -m unittest discover -s tests -v`.

This submission contains the current code, dataset and required evidence. Earlier alternatives and research records are available in Git history at commit [`6abb0ea`](https://github.com/kumawy/ML_end/tree/6abb0ea), including unsuccessful attempts. Cleanup preserves the chosen prediction recipe.
