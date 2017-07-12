# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:12:12 2015

@author: rc, alex
"""
import os
import sys
from time import time
if __name__ == '__main__' and __package__ is None:
    filePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(filePath)

import pdb
import pandas as pd
import numpy as np
import yaml
from copy import deepcopy
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import LeaveOneLabelOut

from preprocessing.aux import getEventNames, delay_preds
from utils.ensembles import createEnsFunc, loadPredictions, getLvl1ModelList, getFastLvl1ModelList

from ensembling.WeightedMean import WeightedMeanClassifier
from ensembling.NeuralNet import NeuralNet
from ensembling.XGB import XGB

from eeg_config import N_EVENTS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_subjects', type=int, action='store', default=2)
parser.add_argument('-fast', type=bool, action='store', default=False)
args, unknown = parser.parse_known_args()
arg_allsubjects = range(1, args.n_subjects + 1)
is_fast = args.fast

# To ignore the warning that: missing __init__.py
# import warnings
# warnings.filterwarnings("ignore", ImportWarning)

def _from_yaml_to_func(method, params):
    """go from yaml to method.

    Need to be here for accesing local variables.
    """
    prm = dict()
    if params is not None:
        for key, val in params.iteritems():
            prm[key] = eval(str(val))
    return eval(method)(**prm)

# ## here read YAML and build models ###
yml = yaml.load(open(sys.argv[1]))

fileName = yml['Meta']['file']
if 'subsample' in yml['Meta']:
    subsample = yml['Meta']['subsample']
else:
    subsample = 1

nbags = yml['Meta']['nbags']
bagsize = yml['Meta']['bagsize']

modelName, modelParams = yml['Model'].iteritems().next()
model_base = _from_yaml_to_func(modelName, modelParams)

ensemble = yml['Model'][modelName]['ensemble']
addSubjectID = True if 'addSubjectID' in yml.keys() else False

mode = sys.argv[2]
if mode == 'val':
    test = False
elif mode == 'test':
    test = True
else:
    raise('Invalid mode. Please specify either val or test')

if test:
    folder = 'test/'
    prefix = 'test_'
else:
    folder = 'val/'
    prefix = 'val_'

print('Running %s in mode %s, will be saved in %s' % (modelName,mode,fileName))

######
cols = getEventNames()

ids = np.load('../infos_test.npy')
subjects_test = ids[:, 1]
series_test = ids[:, 2]
ids = ids[:, 0]
labels = np.load('../infos_val.npy')
subjects = labels[:, -2]
series = labels[:, -1]
labels = labels[:, :-2]

allCols = range(len(cols))

# ## loading prediction ###
files = getFastLvl1ModelList() if is_fast else getLvl1ModelList()

preds_val = OrderedDict()
for f in files:
    loadPredictions(preds_val, f[0], f[1])
# validity check
for m in ensemble:
    assert(m in preds_val)

# ## train/test ###
aggr = createEnsFunc(ensemble)
dataTrain = aggr(preds_val)
preds_val = None

# switch to add subjects
if addSubjectID:
    dataTrain = np.c_[dataTrain, subjects]

np.random.seed(4234521)

report = pd.DataFrame(index=[fileName])
start_time = time()

if test:
    # train the model
    all_models = []
    for k in range(nbags):
        print("Train Bag #%d/%d" % (k+1, nbags))
        model = deepcopy(model_base)
        allsubjects = deepcopy(arg_allsubjects)
        np.random.shuffle(allsubjects)
        ix_subjects = np.sum([subjects==s for s in allsubjects[0:bagsize]], axis=0) != 0
        
        model.mdlNr = k
        model.fit(dataTrain[ix_subjects], labels[ix_subjects])
        all_models.append(model)
    dataTrain = None

    # load test data
    preds_test = OrderedDict()
    for f in files:
        loadPredictions(preds_test, f[0], f[1], test=True)
    dataTest = aggr(preds_test)
    preds_test = None
    # switch to add subjects
    if addSubjectID:
        dataTest = np.c_[dataTest, subjects_test]

    # get predictions
    p = np.zeros((len(ids), N_EVETS))
    for k in range(nbags):
        print("Test Bag #%d" % (k+1))
        model = all_models.pop(0)
        p += model.predict_proba(dataTest) / nbags
    np.save('test/test_%s.npy' % fileName, [p])
else:
    auc_tot = []
    p = np.zeros(labels.shape)
    cv = LeaveOneLabelOut(series)
    for fold, (train, test) in enumerate(cv):
        for k in range(nbags):
            print("Train Bag #%d/%d" % (k+1, nbags))
            allsubjects = deepcopy(arg_allsubjects)
            np.random.shuffle(allsubjects)
            ix_subjects = np.sum([subjects[train]==s for s in allsubjects[0:bagsize]], axis=0) != 0
            model = deepcopy(model_base)
            model.mdlNr = k
            if modelName == 'NeuralNet':
                model.fit(dataTrain[train[ix_subjects]], labels[train[ix_subjects]], dataTrain[test],
                          labels[test])
            else:
                model.fit(dataTrain[train[ix_subjects]], labels[train[ix_subjects]])
            p[test] += model.predict_proba(dataTrain[test]) / nbags
            auc = [roc_auc_score(labels[test][:, col], p[test][:, col])
                   for col in allCols]
            print np.mean(auc)
        auc_tot.append(np.mean(auc))
        print('Fold %d, score: %.5f' % (fold, auc_tot[-1]))
    print('AUC: %.5f' % np.mean(auc_tot))
    report['AUC'] = np.mean(auc_tot)
    np.save('val/val_%s.npy' % fileName, [p])

end_time = time()
report['Time'] = end_time - start_time
report.to_csv("lvl2/report/%s_%s.csv" % (prefix, fileName))
