# Audit of the original study

Audited baseline: `dc0321a6123ff6d567260c8a366ce1a559fe3644`.
Original files under `data/`, `notebooks/`, and `models/` are preserved byte-for-byte.
Machine evidence: [audit.json](../results/audit-20261022/audit.json).
Notebook cell references below are zero-based JSON cell indices.

## Reproduction evidence

The two feature-engineering notebooks reproduce both saved matrices exactly
(maximum absolute numeric difference **0.0**, identical columns and labels).
The processed late-period data match their corresponding cached raw data.
There are no duplicate `(GAME_ID, GAME_EVENT_ID)` pairs in either raw season.

Refitting each of the six saved best configurations and recomputing its original
five-fold CV matches **63 of 72** published metric entries within 1e-8. All 48
accuracy/precision/recall/F1 entries match. Both leading published test AUCs match:
2021–22 Decision Tree **0.6794170203**; 2023–24 Logistic Regression **0.6627942370**.
Nine AUC/CV-AUC entries differ; maximum absolute difference is **0.0008116883**
(2023–24 Random Forest test AUC). See [comparison.csv](../results/legacy-20261022/comparison.csv).

This is a saved-best-configuration reproduction, **not** a rerun of the full
30-candidate search. Original package versions and fitted estimators are not
committed, so exact environment reproduction is unavailable. Differences are
consistent with numerical/library drift, but their exact cause is unproven.
The new environment, warnings, parameters, input hashes, and predictions are saved.

## Leakage and validity findings

| Finding | Evidence | Consequence and response |
|---|---|---|
| Random shots mix past/future and games | `03_feature_engineering.*`, cell 6; 174/217 test games (2021) and 172/219 (2023) also occur in training | Not a forward forecasting estimate; use whole-date temporal splits. This is dependence, not duplicate rows. |
| Training target encoding includes own outcome | Cell 7 maps full training player target means back onto those same training shots | Also carries inner validation labels into CV features. New histories exclude the entire current date and all later dates, with fixed smoothing. The original held-out test labels themselves were not used in this mean. |
| Preprocessing and supervised selection precede CV | Scaling and `SelectKBest` in cells 20/22 occur before modeling cell 8 | Inner-CV validation labels influence selected features, and inner-fold distribution enters scaling. New preprocessing fits inside every fold, with a fixed feature allowlist and no supervised selection. Original outer-test labels were not used for feature selection. |
| Model family selected using test AUC | Modeling cell 18 selects `results_df.iloc[0]` after sorting test AUC | Reported winner has selection bias; follow-up chooses solely by development-fold log loss. |
| Ranking inference does not reproduce training transformations | Modeling cell 20 builds unscaled features, does not reconstruct player history/polynomials, and fills missing selected columns with zero | Published expected probabilities/rankings are not reliable inference from the fitted feature pipeline. All shots, including training shots, are scored. Preserve rankings as historical artifacts; do not present them as validated player dependability. |
| Separate dummy encoding | Feature cell 11 calls `get_dummies(drop_first=True)` independently on train and test, then aligns | Reference-category semantics can differ when category sets differ; new encoder is fitted on training only and tolerates unseen categories. |
| Duplicate polynomial base features | Feature cell 18 appends degree-one columns already present; CSV loading adds `.1` suffixes | Redundant inputs and fragile ranking alignment. New geometry features have unique names. |
| Post-shot event field | `EVENT_TYPE` exactly identifies the target | Original feature code correctly drops it; new pipeline also excludes it. It is not evidence that the original models directly trained on this field. |
| Action annotation availability | `ACTION_TYPE` describes realized shot technique; no feed timing supplied | Not proven target leakage, but cannot assume real-time availability. Excluded from eligible candidates; a separate sensitivity model is logged. |
| No score margin | Raw schemas have clock/period but no score difference | The cohort is last 120 seconds of Q4/OT, including non-close games. It is not a validated close-game clutch cohort. |
| Selected-player population | 27 players in 2021–22, 30 in 2023–24; collection requests a hand-picked list | No league-wide generalization or proof of psychological clutch skill; missing 2022–23 limits history/transfer analysis. |
| EDA includes target/outcome fields | EDA cells 18–22 use all numeric columns, including target, in visualization | PCA/t-SNE separation is not evidence of predictive separability. These embeddings are not passed into the trained classifiers. |

## Source/provenance defects

- `01_data_collection_2021.ipynb` cell 6 writes `shot_data.2023csv` despite a
  2021–22 collection default; the 2023 collection cell also has the missing-dot suffix.
- `02_eda_2023.ipynb` cell 2 reads the 2021 raw file. A literal end-to-end rerun
  would not recreate the stored 2023 processed data. The stored 2023 data itself
  matches the correct 2023 raw file, verified independently.
- Several 2023 modeling plot cells use 2021 filenames and can overwrite 2021 plots.
  The reproduction runner never executes those write/plot cells.
- README described XGBoost/GridSearchCV, but modeling uses LightGBM and
  RandomizedSearchCV; `reqs.txt` is a course checklist, not pip requirements.
  The README's historical body remains intact, with an audit notice and correct
  follow-up instructions at the top. A new pinned `requirements.txt` supports the
  experiment commands; it is not a complete historical notebook environment.

## Prediction contract and remaining limits

New eligible models estimate made-shot probability **at the shot attempt**, with
location, shot type, and clock assumed observed. This is not a before-possession
or season-start forecast. Player features use cached earlier-date attempts only;
no same-game/day outcome enters history. We use all earlier raw shots for prior FG%
and earlier late-period shots for prior late FG%, with a fixed Beta(5,5) prior.
During validation/testing, histories update daily but model weights do not.

No defender, lineup, fatigue, score margin, or live feature feed is available.
Even clean temporal validation on these small selected cohorts cannot establish
league-wide superiority. Holdouts are now inspected: further tuning requires a
new prospective evaluation or additional seasons, not repeated optimization of
these same test scores. No replacement player rankings are asserted in this PR.
