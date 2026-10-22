# Follow-up protocol (fixed before temporal model results)

Original study baseline: dc0321a6123ff6d567260c8a366ce1a559fe3644.
Do not edit original data, notebooks, results, plots, or rankings.

1. Refit all six original estimators using the saved best parameters and saved
   train/test CSVs. Recompute five-fold CV for those configurations and compare
   all six published metrics. This reproduces selected configurations, not the
   entire 30-candidate search. Report mismatches; original package versions are absent.
2. Audit raw/processed equivalence, dates, duplicate shot IDs, same-game overlap,
   target encoding, feature selection, preprocessing, and ranking inference.
3. Retain the original population and outcome: made/missed shots in period >=4
   with <=120 seconds remaining. There is no score margin in these data: call
   this **late-period shooting**, not verified close-game clutch performance.
4. For each season, reserve the last 20% of unique dates as a temporal holdout.
   Use three expanding-window validation folds on earlier dates. Whole dates,
   hence whole games, stay together. Select by mean validation log loss, with
   candidate name as deterministic tie-breaker; never select on test scores.
5. Baselines: training prevalence and distance-only logistic regression.
   Fixed candidate set: context logistic; spatial logistic; spatial plus causal
   player-history logistic (C=0.1 and 1); regularized random forest and LightGBM
   with that same history feature set. Shot-action annotation is a separate
   sensitivity experiment, ineligible for model selection because its availability
   at prediction time is not verified. Preprocessing is fitted within every fold.
6. Prediction time is the shot attempt with location/type/clock observed.
   Histories use only dates strictly before the shot date, from the cached raw
   shots for that season. Same-day outcomes cannot enter history. Validation/test
   history updates daily as past results arrive; model weights remain frozen.
   This is a rolling pregame-information evaluation, not a season-start forecast.
   Fixed Beta(5,5) smoothing avoids full-season priors and supports cold starts.
7. Report accuracy, precision, recall, F1, ROC AUC, average precision, log loss,
   Brier score, and calibration bins. Save split membership, predictions, fold
   metrics, parameters, warnings, package versions, and SHA-256 input/code hashes.
   Report game-cluster paired bootstrap 95% intervals for loss improvements over
   distance-only logistic; do not treat the intervals as proof after model selection.
8. Separately freeze the configuration chosen by 2021–22 validation, fit on all
   2021–22 shots, and test on all 2023–24 late-period shots. 2023–24 player histories
   use earlier dates in that season, with cold starts; 2022–23 is unavailable.
   This is a season-transfer robustness check, not a new untouched dataset: the
   original study and this follow-up both inspect 2023–24.
9. Keep every measured outcome, including failures and negative improvements.
   No automated player-ranking replacement or deployment in this PR.
