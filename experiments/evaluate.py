"""Time-based model selection, holdout evaluation, and season transfer."""
import argparse
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .common import ROOT, SEED, TARGET, dump, manifest, metrics, new_output
from .data import CORE, HISTORY, SPATIAL, assert_ordered, folds, holdout, load_season

# Fixed before evaluating holdouts; action annotation is ineligible for selection.
CANDIDATES = {
    'prevalence': dict(features='distance', model='dummy'),
    'distance_logistic': dict(features='distance', model='logistic', C=1.0),
    'context_logistic': dict(features='context', model='logistic', C=1.0),
    'spatial_logistic': dict(features='spatial', model='logistic', C=1.0),
    'history_logistic_c01': dict(features='history', model='logistic', C=.1),
    'history_logistic_c1': dict(features='history', model='logistic', C=1.0),
    'history_forest': dict(features='history', model='forest'),
    'history_lightgbm': dict(features='history', model='lightgbm'),
    'action_sensitivity': dict(features='action', model='logistic', C=.1),
}


def make_model(name):
    config = CANDIDATES[name]
    feature_set = config['features']
    numeric = ['SHOT_DISTANCE'] if feature_set == 'distance' else list(CORE)
    if feature_set in ('spatial','history','action'):
        numeric += SPATIAL
    if feature_set in ('history','action'):
        numeric += HISTORY
    transforms = [('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                                        ('scale', StandardScaler())]), numeric)]
    if feature_set == 'action':
        transforms.append(('action', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['ACTION_TYPE']))
    preprocessor = ColumnTransformer(transforms, remainder='drop').set_output(transform='pandas')
    if config['model'] == 'dummy':
        estimator = DummyClassifier(strategy='prior')
    elif config['model'] == 'logistic':
        estimator = LogisticRegression(C=config['C'], max_iter=2000, random_state=SEED)
    elif config['model'] == 'forest':
        estimator = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=15,
                                          max_features=.8, random_state=SEED, n_jobs=1)
    else:
        estimator = LGBMClassifier(n_estimators=150, learning_rate=.03, max_depth=3, num_leaves=7,
                                  min_child_samples=30, reg_lambda=5, verbosity=-1, random_state=SEED, n_jobs=1)
    return Pipeline([('preprocess',preprocessor),('model',estimator)])


def select_candidate(rows):
    eligible = pd.DataFrame(rows)
    eligible = eligible[eligible.candidate != 'action_sensitivity']
    ranking = eligible.groupby('candidate').log_loss.mean().reset_index().sort_values(['log_loss','candidate'])
    return ranking.iloc[0].candidate


def prediction_rows(test, probability, protocol, season, candidate):
    frame = test[['GAME_ID','GAME_EVENT_ID','GAME_DATE','PLAYER_ID',TARGET]].copy()
    frame['probability'] = probability
    frame['protocol'], frame['season'], frame['candidate'] = protocol, season, candidate
    return frame


def calibration(test, p):
    frame = pd.DataFrame({'y': np.asarray(test[TARGET]), 'p':p})
    frame['bin'] = np.minimum((frame.p * 10).astype(int), 9)
    return [dict(bin=int(i), n=len(g), mean_probability=float(g.p.mean()), observed_rate=float(g.y.mean()))
            for i,g in frame.groupby('bin')]


def paired_interval(predictions, selected, repetitions=1000):
    """Paired game bootstrap of baseline loss minus selected loss; positive is better."""
    a = predictions[predictions.candidate == selected].set_index(['GAME_ID','GAME_EVENT_ID'])
    b = predictions[predictions.candidate == 'distance_logistic'].set_index(['GAME_ID','GAME_EVENT_ID']).loc[a.index]
    y = a[TARGET].to_numpy()
    def losses(p):
        p = np.clip(np.asarray(p), 1e-15, 1-1e-15)
        return -y*np.log(p)-(1-y)*np.log1p(-p)
    delta = pd.DataFrame({'delta': losses(b.probability)-losses(a.probability)}, index=a.index)
    grouped = delta.groupby(level='GAME_ID').delta.agg(['sum','count']).to_numpy()
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(repetitions):
        draw = grouped[rng.integers(0, len(grouped), size=len(grouped))]
        values.append(draw[:,0].sum()/draw[:,1].sum())
    return dict(selected=selected, reference='distance_logistic', n_games=len(grouped), repetitions=repetitions,
                log_loss_improvement=float(delta.delta.mean()), ci95=np.quantile(values,[.025,.975]).tolist())


def run(out):
    fold_rows, test_rows, prediction_frames, partitions, warnings_log, calibrations = [], [], [], [], [], {}
    selections, season_frames, intervals = {}, {}, {}
    dump(out/'candidate_definitions.json', {n: dict(config=c, estimator=make_model(n).named_steps['model'].get_params()) for n,c in CANDIDATES.items()})

    def fit_predict(name, train, test, stage):
        # Drop target from inputs explicitly, even though the column allowlist also excludes it.
        model = make_model(name)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter('always')
            model.fit(train.drop(columns=TARGET), train[TARGET])
            prob = model.predict_proba(test.drop(columns=TARGET))[:,1]
        warnings_log.extend(dict(stage=stage, candidate=name, category=type(w.message).__name__, message=str(w.message)) for w in caught)
        return prob

    def record_partition(frame, season, split):
        m = frame[['GAME_ID','GAME_EVENT_ID','GAME_DATE']].copy()
        m['season'], m['split'] = season, split
        partitions.append(m)

    for year in (2021, 2023):
        frame = load_season(year)
        season_frames[year] = frame
        dev, test = holdout(frame)
        record_partition(dev, year, 'development')
        record_partition(test, year, 'holdout')
        current_folds = []
        for i, (train, val) in enumerate(folds(dev), start=1):
            record_partition(train, year, f'fold_{i}_train')
            record_partition(val, year, f'fold_{i}_validation')
            for name in CANDIDATES:
                p = fit_predict(name, train, val, f'{year}/fold{i}')
                row = dict(season=year, fold=i, candidate=name, train_n=len(train), validation_n=len(val),
                           train_end=int(train.GAME_DATE.max()), validation_start=int(val.GAME_DATE.min()),
                           validation_end=int(val.GAME_DATE.max()), **metrics(val[TARGET],p))
                fold_rows.append(row)
                current_folds.append(row)
        selected = select_candidate(current_folds)
        selections[str(year)] = selected
        # Save selection before touching the held-out labels for model scoring.
        dump(out/'selection.json', selections)
        print(year, 'validation selected:', selected, flush=True)
        current_predictions = []
        for name in CANDIDATES:
            p = fit_predict(name, dev, test, f'{year}/holdout')
            test_rows.append(dict(protocol='within_season', season=year, candidate=name, selected=name==selected,
                                  train_n=len(dev), test_n=len(test), **metrics(test[TARGET],p)))
            prediction = prediction_rows(test, p, 'within_season', year, name)
            prediction_frames.append(prediction)
            current_predictions.append(prediction)
            calibrations[f'{year}/{name}'] = calibration(test,p)
        intervals[str(year)] = paired_interval(pd.concat(current_predictions), selected)

    train, test = season_frames[2021], season_frames[2023]
    assert_ordered(train, test)
    selected = selections['2021']
    transfer_predictions = []
    for name in dict.fromkeys(['prevalence','distance_logistic',selected]):
        p = fit_predict(name, train, test, 'season_transfer')
        test_rows.append(dict(protocol='season_transfer', season=2023, candidate=name, selected=name==selected,
                              train_n=len(train), test_n=len(test), **metrics(test[TARGET],p)))
        prediction = prediction_rows(test,p,'season_transfer',2023,name)
        prediction_frames.append(prediction)
        transfer_predictions.append(prediction)
        calibrations[f'transfer/{name}'] = calibration(test,p)
    intervals['transfer'] = paired_interval(pd.concat(transfer_predictions),selected)
    pd.DataFrame(fold_rows).to_csv(out/'validation.csv',index=False)
    pd.DataFrame(test_rows).to_csv(out/'metrics.csv',index=False)
    pd.concat(prediction_frames).to_csv(out/'predictions.csv',index=False)
    pd.concat(partitions).to_csv(out/'splits.csv',index=False)
    dump(out/'calibration.json',calibrations)
    dump(out/'paired_intervals.json',intervals)
    dump(out/'warnings.json',warnings_log)
    manifest(out,'python -m experiments.evaluate --output '+str(out))
    print(pd.DataFrame(test_rows)[['protocol','season','candidate','selected','auc','log_loss','brier']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(new_output(parser.parse_args().output))
