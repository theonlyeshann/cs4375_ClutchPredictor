"""Regression tests for leakage boundaries, split integrity, and inference."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from experiments.common import ROOT, TARGET, new_output
from experiments.data import HISTORY, KEY, assert_ordered, build_features, folds, holdout, load_season, validate_raw
from experiments.evaluate import CANDIDATES, make_model, paired_interval, prediction_rows, select_candidate
from experiments.reproduce import saved_parameters


@pytest.fixture
def raw():
    return pd.DataFrame([
        dict(GAME_ID=100+day, GAME_EVENT_ID=event, PLAYER_ID=player, GAME_DATE=20230101+day,
             SHOT_MADE_FLAG=made, SHOT_DISTANCE=15, LOC_X=20, LOC_Y=100, PERIOD=4,
             MINUTES_REMAINING=1, SECONDS_REMAINING=20, SHOT_TYPE='2PT Field Goal',
             ACTION_TYPE='Jump Shot')
        for day in range(12) for event,player,made in [(1,10,day%2),(2,10,(day+1)%2),(3,20,day%2)]])


def test_same_day_and_future_labels_cannot_change_features(raw):
    before = build_features(raw)
    changed = raw.copy()
    changed.loc[changed.GAME_DATE >= 20230106,TARGET] = 1-changed.loc[changed.GAME_DATE >= 20230106,TARGET]
    after = build_features(changed)
    columns = KEY+HISTORY
    assert_frame_equal(before.loc[before.GAME_DATE <= 20230106,columns],after.loc[after.GAME_DATE <= 20230106,columns])
    assert not before.loc[before.GAME_DATE==20230107,HISTORY].equals(after.loc[after.GAME_DATE==20230107,HISTORY])


def test_history_is_previous_day_with_fixed_cold_start(raw):
    result = build_features(raw)
    first = result[result.GAME_DATE==20230101]
    np.testing.assert_array_equal(first.prior_fg,.5)
    np.testing.assert_array_equal(first.log_prior_attempts,0)
    day2 = result[(result.GAME_DATE==20230102)&(result.PLAYER_ID==10)]
    np.testing.assert_allclose(day2.prior_fg,6/12)
    np.testing.assert_allclose(day2.log_prior_attempts,np.log1p(2))


def test_future_rows_and_input_order_do_not_change_past(raw):
    full = build_features(raw)
    prefix = build_features(raw[raw.GAME_DATE <= 20230105])
    assert_frame_equal(full[full.GAME_DATE <= 20230105].reset_index(drop=True),prefix)
    assert_frame_equal(full,build_features(raw.sample(frac=1,random_state=1)))


def test_clutch_boundary_and_overtime(raw):
    raw = raw.iloc[:5].copy()
    raw['PERIOD'] = [3,4,4,5,4]
    raw['MINUTES_REMAINING'] = [1,2,2,2,0]
    raw['SECONDS_REMAINING'] = [0,0,1,0,0]
    result = build_features(raw)
    assert set(map(tuple,result[KEY].values)) == set(map(tuple,raw.iloc[[1,3,4]][KEY].values))


@pytest.mark.parametrize('corruption',['duplicate','label','missing','date','infinite','game_date','clock','shot_type'])
def test_invalid_data_rejected(raw,corruption):
    if corruption=='duplicate': raw=pd.concat([raw,raw.iloc[:1]])
    if corruption=='label': raw.loc[0,TARGET]=2
    if corruption=='missing': raw.loc[0,'PLAYER_ID']=np.nan
    if corruption=='date': raw.loc[0,'GAME_DATE']=20230230
    if corruption=='infinite':
        raw['LOC_Y']=raw.LOC_Y.astype(float)
        raw.loc[0,'LOC_Y']=np.inf
    if corruption=='game_date': raw.loc[0,'GAME_DATE']=20230102
    if corruption=='clock': raw.loc[0,'SECONDS_REMAINING']=60
    if corruption=='shot_type': raw.loc[0,'SHOT_TYPE']='Unknown'
    with pytest.raises(ValueError): validate_raw(raw)


def test_whole_date_splits(raw):
    frame=build_features(raw)
    dev,test=holdout(frame)
    assert len(dev)+len(test)==len(frame)
    assert_ordered(dev,test)
    for train,val in folds(dev):
        assert_ordered(train,val)
        assert set(train.GAME_DATE).isdisjoint(val.GAME_DATE)
    with pytest.raises(ValueError): assert_ordered(test,dev)
    with pytest.raises(ValueError): assert_ordered(dev,dev)


@pytest.mark.parametrize('year',[2021,2023])
def test_real_data_split_and_saved_parameters(year):
    frame=load_season(year)
    dev,test=holdout(frame)
    assert_ordered(dev,test)
    assert len(saved_parameters(year))==6
    assert not frame.duplicated(KEY).any()


@pytest.mark.parametrize('name',list(CANDIDATES))
def test_prediction_ignores_target_and_identifiers(raw,name):
    frame=build_features(raw)
    dev,test=holdout(frame)
    model=make_model(name).fit(dev.drop(columns=TARGET),dev[TARGET])
    expected=model.predict_proba(test)[:,1]
    changed=test.copy()
    changed[TARGET]=1-changed[TARGET]
    changed['EVENT_TYPE']='Made Shot'
    changed['GAME_EVENT_ID']=99999
    changed['PLAYER_ID']=99999
    changed['PLAYER_NAME']='Unseen player'
    # Histories were already computed from historical IDs; model cannot use IDs.
    np.testing.assert_allclose(expected,model.predict_proba(changed)[:,1])
    assert np.isfinite(expected).all()
    assert ((expected>=0)&(expected<=1)).all()
    if name=='action_sensitivity':
        changed['ACTION_TYPE']='Unseen annotation'
        assert np.isfinite(model.predict_proba(changed)).all()


def test_preprocessing_fits_training_only(raw):
    frame=build_features(raw)
    dev,test=holdout(frame)
    dev=dev.copy()
    dev['SHOT_DISTANCE']=np.arange(len(dev))
    model=make_model('distance_logistic').fit(dev,dev[TARGET])
    scaler=model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale']
    np.testing.assert_allclose(scaler.mean_,dev[['SHOT_DISTANCE']].mean())
    old_mean=scaler.mean_.copy()
    test=test.copy(); test['SHOT_DISTANCE']=999999
    model.predict_proba(test)
    np.testing.assert_array_equal(old_mean,scaler.mean_)


def test_selection_uses_validation_and_excludes_action():
    rows=[dict(candidate='action_sensitivity',log_loss=.01),dict(candidate='context_logistic',log_loss=.7),
          dict(candidate='distance_logistic',log_loss=.6)]
    assert select_candidate(rows)=='distance_logistic'


def test_bootstrap_identical_model_is_zero(raw):
    frame=build_features(raw)
    p=np.full(len(frame),.5)
    rows=pd.concat([prediction_rows(frame,p,'test',2021,n) for n in ['distance_logistic','context_logistic']])
    result=paired_interval(rows,'context_logistic',repetitions=50)
    assert result['log_loss_improvement']==0
    assert result['ci95']==[0,0]


def test_outputs_never_overwrite(tmp_path):
    new_output(tmp_path/'run')
    with pytest.raises(FileExistsError): new_output(tmp_path/'run')
    with pytest.raises(ValueError): new_output(ROOT/'data'/'new-results')
