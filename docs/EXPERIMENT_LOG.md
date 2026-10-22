# Experiment log — 2026-10-22

These are measured results, including weak and negative findings. The original
study is preserved. Definitions and selection rules were written in
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) before temporal evaluation.
Use [RUNNING.md](RUNNING.md) to reproduce. Each run directory is immutable.

## Runs and decisions

| Run | Purpose | Outcome / decision |
|---|---|---|
| [audit-20261022](../results/audit-20261022/) | Original raw/processed/split/feature provenance | Both engineered train/test matrices reproduce exactly; raw and processed cohorts agree. Random split shares 174 / 172 games. See [audit](AUDIT.md). |
| [legacy-20261022](../results/legacy-20261022/) | Refit saved best configurations; original 5-fold CV | 63/72 metric entries match within 1e-8; all 48 classification metric entries match. Maximum absolute AUC/CV-AUC discrepancy 0.000812. No original environment lock exists. |
| [temporal-20261022](../results/temporal-20261022/) | Nine candidates, 3 expanding folds, two temporal holdouts, season transfer | Random forest chosen by validation log loss in both seasons. Useful 2021–22 improvement; no clear 2023–24 or transfer advantage over distance baseline. |
| [verification-20261022](../results/verification-20261022/) | Leakage regression tests and final repeatability verification | 28 tests pass without warnings. Final temporal rerun agrees within 1e-12; maximum CSV numeric difference 2.22e-16. No original files changed. |

The initial runs were made on the feature branch from base
`dc0321a6123ff6d567260c8a366ce1a559fe3644`; manifests record dirty working-tree
state and content hashes. Subsequent cleanup retains pandas feature names through
the inference pipeline, removes a test-fixture warning, and supports absolute CLI
output paths. It changes neither feature definitions nor model-selection rules.

## Original evaluation reproduction

| Season | Model | Published test AUC | Refit test AUC |
|---|---|---:|---:|
| 2021–22 | Decision Tree | 0.679417 | 0.679417 |
| 2021–22 | MLP | 0.657862 | 0.657862 |
| 2021–22 | Random Forest | 0.656690 | 0.657025 |
| 2021–22 | Logistic Regression | 0.654121 | 0.654121 |
| 2021–22 | LightGBM | 0.651748 | 0.651748 |
| 2021–22 | SVM | 0.651552 | 0.651385 |
| 2023–24 | Logistic Regression | 0.662794 | 0.662794 |
| 2023–24 | LightGBM | 0.662464 | 0.662464 |
| 2023–24 | Decision Tree | 0.662464 | 0.662464 |
| 2023–24 | Random Forest | 0.660765 | 0.661577 |
| 2023–24 | MLP | 0.649046 | 0.649046 |
| 2023–24 | SVM | 0.634233 | 0.634334 |

Full classification and CV differences are in `legacy-20261022/comparison.csv`.
The saved best parameter configurations were used, not a new 30-candidate search.
Reproducing flawed evaluation does not validate its use for future prediction.

## Temporal splits

| Season | Development dates | Development shots | Holdout dates | Holdout shots / games |
|---|---|---:|---|---:|
| 2021–22 | 2021-10-19 → 2022-03-05 | 1,078 | 2022-03-06 → 2022-04-09 | 265 / 101 |
| 2023–24 | 2023-10-24 → 2024-03-09 | 1,155 | 2024-03-10 → 2024-04-12 | 252 / 106 |

Last 20% of unique observed late-shot dates form each holdout. Inner folds expand
on the preceding dates. Every game remains in one partition per fold. History
uses all earlier-date raw shots; held-out histories update as days finish while
model weights remain fixed. Full shot-key membership: `splits.csv`.

## Candidate ablation and modelling results

Lower log loss is better. Model selection uses only mean development-fold loss.
Holdout columns are diagnostic; they do not change the selected model.

| Candidate | 2021 validation loss | 2021 holdout loss | 2023 validation loss | 2023 holdout loss |
|---|---:|---:|---:|---:|
| Training prevalence | 0.691810 | 0.685721 | 0.689578 | 0.685608 |
| Distance logistic | 0.642649 | 0.634996 | 0.642184 | 0.626789 |
| Context logistic | 0.652938 | 0.630798 | 0.643111 | 0.624860 |
| Spatial logistic | 0.658688 | 0.625701 | 0.641118 | 0.629965 |
| History logistic, C=0.1 | 0.661339 | 0.625710 | 0.644779 | 0.626765 |
| History logistic, C=1 | 0.668864 | 0.624069 | 0.639472 | 0.630617 |
| **History random forest** | **0.637765** | **0.606276** | **0.638855** | **0.625333** |
| History LightGBM | 0.639198 | 0.602242 | 0.644308 | 0.629240 |
| Action annotation sensitivity, ineligible | 0.656508 | 0.613039 | 0.638919 | 0.623776 |

- Spatial/history features do not consistently improve linear models.
- LightGBM has lower 2021 holdout loss than the selected forest, but it was not the
  validation winner. Switching to it using this holdout would be test-set tuning.
- Action annotations have uncertain prediction-time availability and are excluded
  from model selection, even when their diagnostic holdout result looks better.
- These ablations compare feature sets for logistic regression. The forest result
  does not isolate the causal contribution of history versus nonlinear modelling.

## Selected-model evidence

| Evaluation | Model | Test AUC | Log loss | Brier | Accuracy |
|---|---|---:|---:|---:|---:|
| 2021–22 temporal | Distance logistic | 0.695557 | 0.634996 | 0.222535 | 0.626415 |
| 2021–22 temporal | Selected forest | 0.737619 | 0.606276 | 0.208587 | 0.664151 |
| 2023–24 temporal | Distance logistic | 0.695644 | 0.626789 | 0.218096 | 0.642857 |
| 2023–24 temporal | Selected forest | 0.694200 | 0.625333 | 0.216475 | 0.678571 |
| 2021→2023 transfer | Distance logistic | 0.687955 | 0.638703 | 0.223608 | 0.642502 |
| 2021→2023 transfer | 2021-selected forest | 0.681477 | 0.634255 | 0.221050 | 0.651741 |

Transfer fits on all 1,343 late-period shots from 2021–22 and scores all 1,407
from 2023–24. Its configuration comes solely from 2021–22 development validation.
2023 histories cold-start with the fixed prior and update using earlier dates in
2023–24; 2022–23 is absent. This is a robustness check on previously studied data,
not a claim of a fully untouched external dataset.

Game-cluster paired bootstrap (1,000 draws, seed 42), baseline log loss minus
selected forest log loss; positive means improvement:

| Evaluation | Improvement | 95% interval |
|---|---:|---:|
| 2021–22 | 0.028720 | [0.005733, 0.054208] |
| 2023–24 | 0.001456 | [-0.026042, 0.029740] |
| Transfer | 0.004449 | [-0.006435, 0.015667] |

These intervals cluster shots within games but do not capture shared-player or
cross-game temporal dependence, training uncertainty, or all model-selection
uncertainty. They are descriptive evidence, not a guarantee of improvement.
There is no convincing consistent win in 2023–24 or transfer; the distance model
remains an important benchmark. All metrics, predictions, calibration bins,
parameters, warning messages, and data/code hashes are retained in run artifacts.

## Review decision

Recommended: accept the reproducibility/validation infrastructure and honest
results as a research improvement. Do not claim universal model superiority,
replace the original player rankings, or compare new AUCs against old random-split
AUCs as a like-for-like gain. Before further tuning, acquire a genuinely new
season and verified score-margin/feature-availability data. This PR does not
merge, deploy, or modify the historical study automatically.

Repeatability note: a strict byte-equality check initially failed for four files
after the feature-name cleanup. Investigation found only floating-point changes
(maximum CSV numeric difference 2.22e-16). All eight checked evidence files match
within absolute tolerance 1e-12; selections, split membership, and bootstrap
intervals are byte-identical. This failed strict check and the subsequent numeric
comparison are retained in `verification-20261022/repeatability.json`.
