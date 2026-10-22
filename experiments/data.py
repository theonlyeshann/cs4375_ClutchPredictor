"""Validated cached data and strictly previous-date player histories."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .common import ROOT, TARGET

KEY = ['GAME_ID', 'GAME_EVENT_ID']
HISTORY = ['prior_fg', 'prior_late_fg', 'log_prior_attempts', 'log_prior_late_attempts']
CORE = ['SHOT_DISTANCE', 'TIME_REMAINING', 'PERIOD', 'is_three']
SPATIAL = ['LOC_X', 'LOC_Y', 'abs_x', 'distance_squared', 'angle', 'under_10_seconds']


def validate_raw(raw):
    required = KEY + ['PLAYER_ID', 'GAME_DATE', TARGET, 'SHOT_DISTANCE', 'LOC_X', 'LOC_Y',
                      'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING', 'SHOT_TYPE', 'ACTION_TYPE']
    if not set(required) <= set(raw):
        raise ValueError('Missing required raw columns')
    if raw[required].isna().any().any():
        raise ValueError('Missing raw values')
    numeric = [c for c in required if c not in ('SHOT_TYPE', 'ACTION_TYPE')]
    if not np.isfinite(raw[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite raw values')
    if raw.duplicated(KEY).any():
        raise ValueError('Duplicate shot IDs')
    if not raw[TARGET].isin([0, 1]).all():
        raise ValueError('Target must be binary')
    if raw.groupby('GAME_ID').GAME_DATE.nunique().max() != 1:
        raise ValueError('Game spans multiple dates')
    pd.to_datetime(raw.GAME_DATE.astype(str), format='%Y%m%d', errors='raise')
    if not raw.SHOT_TYPE.isin(['2PT Field Goal', '3PT Field Goal']).all():
        raise ValueError('Unexpected shot type')
    if (raw.PERIOD < 1).any() or not raw.SECONDS_REMAINING.between(0, 59).all() or not raw.MINUTES_REMAINING.between(0, 12).all():
        raise ValueError('Invalid game clock')


def build_features(raw):
    """Deterministic features; histories update only after a whole date has ended.

    Beta(5,5) is fixed in advance. No season-wide target mean is used. Computing
    these once is safe across temporal folds because future rows cannot change
    any earlier feature; mutation tests enforce that property.
    """
    validate_raw(raw)
    df = raw.copy()
    df['TIME_REMAINING'] = 60 * df.MINUTES_REMAINING + df.SECONDS_REMAINING
    late = (df.PERIOD >= 4) & (df.TIME_REMAINING <= 120)
    df['_late_attempt'] = late.astype(int)
    df['_late_make'] = df[TARGET] * df['_late_attempt']
    daily = df.groupby(['PLAYER_ID', 'GAME_DATE'], sort=True).agg(
        makes=(TARGET, 'sum'), attempts=(TARGET, 'size'),
        late_makes=('_late_make', 'sum'), late_attempts=('_late_attempt', 'sum'))
    prior = daily.groupby(level=0).cumsum() - daily
    daily['prior_fg'] = (prior.makes + 5) / (prior.attempts + 10)
    daily['prior_late_fg'] = (prior.late_makes + 5) / (prior.late_attempts + 10)
    daily['log_prior_attempts'] = np.log1p(prior.attempts)
    daily['log_prior_late_attempts'] = np.log1p(prior.late_attempts)
    df = df.join(daily[HISTORY], on=['PLAYER_ID', 'GAME_DATE'], validate='many_to_one')
    df['is_three'] = (df.SHOT_TYPE == '3PT Field Goal').astype(int)
    df['abs_x'] = df.LOC_X.abs()
    df['distance_squared'] = df.SHOT_DISTANCE ** 2
    df['angle'] = np.arctan2(df.LOC_X.abs(), df.LOC_Y)
    df['under_10_seconds'] = (df.TIME_REMAINING <= 10).astype(int)
    return df.loc[late].drop(columns=['_late_attempt','_late_make']).sort_values(
        ['GAME_DATE', 'GAME_ID', 'GAME_EVENT_ID']).reset_index(drop=True)


def load_season(year):
    return build_features(pd.read_csv(ROOT/f'data/raw/shot_data.{year}.csv'))


def assert_ordered(train, test):
    if train.empty or test.empty:
        raise ValueError('Empty temporal partition')
    if train.GAME_DATE.max() >= test.GAME_DATE.min():
        raise ValueError('Dates overlap or run backwards')
    if set(train.GAME_ID) & set(test.GAME_ID):
        raise ValueError('Games overlap')


def holdout(df):
    dates = np.sort(df.GAME_DATE.unique())
    cutoff = int(len(dates) * .8)
    train, test = df[df.GAME_DATE.isin(dates[:cutoff])], df[df.GAME_DATE.isin(dates[cutoff:])]
    assert_ordered(train, test)
    return train, test


def folds(df):
    dates = np.sort(df.GAME_DATE.unique())
    for train_dates, val_dates in TimeSeriesSplit(n_splits=3).split(dates):
        train, val = df[df.GAME_DATE.isin(dates[train_dates])], df[df.GAME_DATE.isin(dates[val_dates])]
        assert_ordered(train, val)
        yield train, val
