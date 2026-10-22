"""Machine-checkable data/provenance audit; detailed interpretation in docs/AUDIT.md."""
import argparse
import contextlib
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import ROOT, TARGET, dump, manifest, new_output
from .data import KEY, validate_raw


def run(out):
    audit = {}
    for year in (2021, 2023):
        raw = pd.read_csv(ROOT/f'data/raw/shot_data.{year}.csv')
        validate_raw(raw)
        saved = pd.read_csv(ROOT/f'data/processed/clutch_shot_data.{year}.csv')
        late = raw[(raw.PERIOD >= 4) & ((raw.MINUTES_REMAINING * 60 + raw.SECONDS_REMAINING) <= 120)]
        a = late.sort_values(KEY)[raw.columns].reset_index(drop=True)
        b = saved.sort_values(KEY)[raw.columns].reset_index(drop=True)
        tr, te = train_test_split(saved, test_size=.2, random_state=42, stratify=saved[TARGET])
        engineered = {s: pd.read_csv(ROOT/f'data/processed/engineered_clutch_{s}.{year}.csv') for s in ('train','test')}
        record = dict(raw_rows=len(raw), late_rows=len(late), players=raw.PLAYER_ID.nunique(),
                      first_date=int(raw.GAME_DATE.min()), last_date=int(raw.GAME_DATE.max()),
                      duplicate_shot_ids=int(raw.duplicated(KEY).sum()), processed_matches_raw=a.equals(b),
                      random_train_rows=len(tr), random_test_rows=len(te),
                      shared_games=len(set(tr.GAME_ID) & set(te.GAME_ID)),
                      test_games=te.GAME_ID.nunique(), train_max_date=int(tr.GAME_DATE.max()),
                      test_min_date=int(te.GAME_DATE.min()),
                      saved_train_labels_match_split=np.array_equal(tr[TARGET],engineered['train'][TARGET]),
                      saved_test_labels_match_split=np.array_equal(te[TARGET],engineered['test'][TARGET]),
                      event_type_encodes_target=bool(((raw.EVENT_TYPE == 'Made Shot').astype(int) == raw[TARGET]).all()))
        # Execute ONLY reviewed feature cells, excluding the write cell, against
        # original input. No notebook model/ranking/collection writes are run.
        notebook = json.loads((ROOT/f'notebooks/03_feature_engineering.{year}.ipynb').read_text())
        env, stdout = {}, io.StringIO()
        old_cwd = os.getcwd()
        cell_index = None
        try:
            os.chdir(ROOT/'notebooks')
            with contextlib.redirect_stdout(stdout), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                for cell_index, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] == 'code' and cell_index != 24:
                        exec(compile(''.join(cell['source']), f'{year}:cell{cell_index}', 'exec'), env)
            record['feature_reexecution'] = 'completed'
            record['feature_warnings'] = sorted(set(str(w.message) for w in caught))
            for split in ('train','test'):
                # CSV serializing duplicates adds .1 suffixes on re-read.
                rebuilt = pd.DataFrame(env[f'X_{split}_selected'], columns=env['selected_features'])
                rebuilt[TARGET] = env[f'y_{split}'].values
                rebuilt = pd.read_csv(io.StringIO(rebuilt.to_csv(index=False)))
                record[f'rebuilt_{split}_columns_match'] = rebuilt.columns.equals(engineered[split].columns)
                record[f'rebuilt_{split}_max_abs_difference'] = float(np.max(np.abs(rebuilt.to_numpy()-engineered[split].to_numpy()))) if rebuilt.columns.equals(engineered[split].columns) else None
        except Exception as error:
            record['feature_reexecution'] = dict(failed_cell=cell_index, error=type(error).__name__, message=str(error))
        finally:
            os.chdir(old_cwd)
        (out/f'feature-reexecution-{year}.txt').write_text(stdout.getvalue())
        audit[str(year)] = record
    dump(out/'audit.json', audit)
    manifest(out, 'python -m experiments.audit --output '+str(out))
    print(json.dumps(audit, indent=2, default=int))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(new_output(parser.parse_args().output))
