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

import numpy as np
import pandas as pd
import pdb
import yaml
from copy import deepcopy
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import LeaveOneLabelOut

from preprocessing.aux import getEventNames, delay_preds
from utils.ensembles import createEnsFunc, loadPredictions, getLvl1ModelList

from ensembling.WeightedMean import WeightedMeanClassifier
from ensembling.NeuralNet import NeuralNet
from ensembling.XGB import XGB

from eeg_config import N_EVENTS

# import warnings
# warnings.filterwarnings("ignore",category=DeprecationWarning)

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

modelName, modelParams = yml['Model'].iteritems().next()
model_base = _from_yaml_to_func(modelName, modelParams)
ensemble = yml['Model'][modelName]['ensemble']
    
nbags = yml['Meta']['nbags']
bagsize = yml['Meta']['bagsize']

addSubjectID = True if 'addSubjectID' in yml.keys() else False

if 'seed' in yml['Meta']:
    seed = yml['Meta']['seed']
else:    
    seed = 4234521

mode = sys.argv[2]
if mode == 'val':
    test = False
elif mode == 'test':
    test = True
else:
    raise('Invalid mode. Please specify either val or test')

print('Running %s in mode %s, will be saved in %s' % (modelName,mode,fileName))

######
cols = getEventNames()

labels = np.load('../infos_val.npy')
subjects = labels[:, -2]
series = labels[:, -1]
labels = labels[:, :-2]

allCols = range(len(cols))

# ## loading prediction ###
files = getLvl1ModelList()

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

np.random.seed(seed)

report = pd.DataFrame(index=[fileName])
start_time = time()

auc_tot = []
p = np.zeros(labels.shape)
cv = LeaveOneLabelOut(series)
for fold, (train, test) in enumerate(cv):
    allmodels = []
    allbags = []
    for k in range(nbags):
        print("Train Bag #%d/%d" % (k+1, nbags))
        model = deepcopy(model_base)
        bag = np.arange(len(ensemble))
        np.random.shuffle(bag)
        bag = bag[0:bagsize]
        selected_model = [i in bag for i in np.arange(len(ensemble))]
        
        model.ensemble = list(np.array(ensemble)[np.where(selected_model)[0]])
        
        selected_model = np.repeat(selected_model,N_EVENTS)
        allbags.append(selected_model)
        model.mdlNr = k
        if modelName == 'NeuralNet':
            model.fit(dataTrain[train][:, selected_model], labels[train], dataTrain[test][:, selected_model], labels[test])
        else:
            model.fit(dataTrain[train][:, selected_model], labels[train])
        p[test] += model.predict_proba(dataTrain[test][:, selected_model]) / nbags
        auc = [roc_auc_score(labels[test][:, col], p[test][:, col])
                for col in allCols]
        print np.mean(auc)
    auc_tot.append(np.mean(auc))
    print('Fold %d, score: %.5f' % (fold, auc_tot[-1]))
end_time = time()
report['Time'] = end_time - start_time
print('AUC: %.5f' % np.mean(auc_tot))
report['AUC'] = np.mean(auc_tot)
report.to_csv("report/%s_%s.csv" % (prefix, fileName))
np.save('val/val_%s.npy' % fileName, [p])
