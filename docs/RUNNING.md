# Run the follow-up experiments

From the repository root, using Python **3.12**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m pytest -q
python -m experiments.audit --output results/my-audit
python -m experiments.reproduce --output results/my-legacy
python -m experiments.evaluate --output results/my-temporal
```

On Windows use `.venv\Scripts\activate`, and set the thread variables with your
shell's environment-variable syntax. CPU-only; no NBA API calls or credentials.
Every output directory must be new. Existing runs are never overwritten and
output inside `data/`, `notebooks/`, or `models/` is rejected.
Absolute output paths are also supported.

- `audit`: validates shot IDs and raw/processed provenance, measures original
  random-split overlap, and re-executes the reviewed feature notebook cells in
  memory, excluding write cells. It records failures rather than silently fixing
  historical code. Do not use this runner on unreviewed third-party notebooks.
- `reproduce`: parses saved best parameters as literals, refits all six original
  model types, and recomputes original five-fold CV. It records each published
  metric, measured value, difference, and 1e-8 match status. It deliberately
  reproduces the historical flawed protocol for comparison, not for deployment.
- `evaluate`: evaluates nine predefined candidates on three expanding folds per
  season. It writes the validation selection before evaluating the date holdout,
  then performs the 2021-to-2023 season-transfer check.

`manifest.json` records package versions, actual command, code/data/notebook
SHA-256 hashes, base HEAD, working-tree state, platform, and seed. Early run
manifests correctly show a dirty development tree; hashes identify actual bytes.
The model-search definitions and complete estimator parameters are recorded.
Pipeline estimators are refitted on each run; no untrusted pickle is loaded.

The pinned environment covers the follow-up CLI and tests. Historical notebook
EDA/collection additionally needs plotting/Jupyter/nba_api packages, and the
original end-to-end notebook path has documented defects. Use the cached-data
commands above for reproducible evidence.

Tests specifically cover future/same-day target mutation, order invariance,
missing/nonfinite/duplicate/invalid data, clock boundaries, disjoint games/dates,
train-only scaling, target/identifier exclusion, unseen action categories,
validation-only model selection, bootstrap pairing, and write protection.
