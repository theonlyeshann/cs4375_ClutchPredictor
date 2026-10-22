"""Shared provenance and metrics for immutable experiment directories."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'SHOT_MADE_FLAG'
SEED = 42


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False) + '\n')


def metrics(y, p, predicted=None):
    predicted = (np.asarray(p) >= .5).astype(int) if predicted is None else predicted
    return dict(accuracy=accuracy_score(y, predicted), precision=precision_score(y, predicted, zero_division=0),
                recall=recall_score(y, predicted, zero_division=0), f1=f1_score(y, predicted, zero_division=0),
                auc=roc_auc_score(y, p) if len(set(y)) == 2 else None,
                average_precision=average_precision_score(y, p), log_loss=log_loss(y, p, labels=[0, 1]),
                brier=brier_score_loss(y, p))


def manifest(out, command):
    files = sorted(set(ROOT.glob('data/**/*.csv')) | set(ROOT.glob('notebooks/*.ipynb')) |
                   set(ROOT.glob('experiments/*.py')) | set(ROOT.glob('tests/*.py')) |
                   {ROOT/'requirements.txt', ROOT/'docs/EXPERIMENT_PROTOCOL.md'})
    dump(out/'manifest.json', dict(command=command, seed=SEED, created_utc=datetime.now(timezone.utc).isoformat(),
         python=platform.python_version(), platform=platform.platform(),
         git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
         packages={n: importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','lightgbm','pytest']},
         sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))


def new_output(path):
    out = Path(path).resolve()
    # Never write into preserved original study directories.
    for protected in ('data', 'notebooks', 'models'):
        if out == ROOT/protected or ROOT/protected in out.parents:
            raise ValueError('Output must not be inside the original study')
    out.mkdir(parents=True, exist_ok=False)
    return out
