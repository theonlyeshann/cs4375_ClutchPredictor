"""Refit saved best configurations without executing notebook write/plot cells."""
import argparse
import ast
import json
import re
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .common import ROOT, SEED, TARGET, dump, manifest, metrics, new_output


def saved_parameters(year):
    nb = json.loads((ROOT/f'notebooks/04_modeling.{year}.ipynb').read_text())
    output = ''.join(''.join(o.get('text', [])) for o in nb['cells'][8]['outputs'])
    found = re.findall(r'Tuning (.*?)\.\.\.\s+Best CV AUC: .*? \| Params: (\{[^\n]+\})', output)
    # Only parse literals; do not execute stored notebook output.
    return {name: ast.literal_eval(re.sub(r'np\.float64\(([^)]+)\)', r'\1', params)) for name, params in found}


def run(out):
    rows, predictions, params_log, warning_log = [], [], {}, []
    for year in (2021, 2023):
        train = pd.read_csv(ROOT/f'data/processed/engineered_clutch_train.{year}.csv')
        test = pd.read_csv(ROOT/f'data/processed/engineered_clutch_test.{year}.csv')
        published = pd.read_csv(ROOT/f'data/processed/model_results.{year}.csv').set_index('Model')
        estimators = {'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
            'SVM': SVC(probability=True, random_state=SEED),
            'Decision Tree': DecisionTreeClassifier(random_state=SEED),
            'Random Forest': RandomForestClassifier(random_state=SEED, n_jobs=1),
            'LightGBM': LGBMClassifier(random_state=SEED, verbosity=-1, n_jobs=1),
            'MLP': MLPClassifier(max_iter=500, random_state=SEED, early_stopping=True)}
        params = saved_parameters(year)
        if set(params) != set(estimators):
            raise ValueError('Incomplete saved best parameter output')
        params_log[str(year)] = params
        x, y = train.drop(columns=TARGET), train[TARGET]
        xt, yt = test.drop(columns=TARGET), test[TARGET]
        for name, estimator in estimators.items():
            model = clone(estimator).set_params(**params[name])
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter('always')
                model.fit(x, y)
                p, pred = model.predict_proba(xt)[:, 1], model.predict(xt)
                cv = cross_val_score(clone(model), x, y, scoring='roc_auc',
                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1)
            warning_log.extend(dict(season=year, model=name, category=type(w.message).__name__, message=str(w.message)) for w in caught)
            result = metrics(yt, p, pred)
            for metric, original in [('accuracy','Accuracy'),('precision','Precision'),('recall','Recall'),('f1','F1'),('auc','AUC'),('cv_auc','CV_AUC')]:
                actual = float(cv.mean()) if metric == 'cv_auc' else result[metric]
                expected = float(published.loc[name, original])
                rows.append(dict(season=year, model=name, metric=metric, published=expected,
                                 reproduced=actual, delta=actual-expected, matches_1e_8=abs(actual-expected) <= 1e-8))
            predictions.extend(dict(season=year, model=name, saved_test_row=i, y=int(t), probability=float(prob), predicted=int(label))
                               for i,(t,prob,label) in enumerate(zip(yt,p,pred)))
            print(year, name, 'AUC', round(result['auc'], 6), 'CV', round(float(cv.mean()), 6), flush=True)
    pd.DataFrame(rows).to_csv(out/'comparison.csv', index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv', index=False)
    dump(out/'parameters.json', params_log)
    dump(out/'warnings.json', warning_log)
    manifest(out, 'python -m experiments.reproduce --output '+str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(new_output(parser.parse_args().output))
