# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:56:55 2015.

@author: alex, rc

This script contain code to generate lvl1 model prediction.
usage : python genPreds.py model_name mode
with mode = val for validation and val = test for test.

This script will read the model description from the yaml file, load
dependencies, create preprocessing and classification pipeline and apply them
on raw data independently on each subjects.

This script support caching of preprocessed data, in order to allow reuse of
preprocessing pipeline across model.
"""
import os
import sys
if __name__ == '__main__' and __package__ is None:
    filePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(filePath)

import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
import yaml
# from sklearn.pipeline import make_pipeline, Pipeline
from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker

from sklearn.metrics import roc_auc_score

from preprocessing.aux import getEventNames, load_raw_data

from multiprocessing import Pool

from read_adapter import subjects_path_list
from eeg_config import subjects
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_subjects', type=int, action='store', default=2)
args, unknown = parser.parse_known_args()
subjects = range(1, args.n_subjects + 1)

cols = getEventNames()


def _from_yaml_to_func(method, params):
    """go from yaml to method.

    Need to be here for accesing local variables.
    """
    prm = dict()
    if params is not None:
        for key, val in params.iteritems():
            prm[key] = eval(str(val))
    return eval(method)(**prm)


def doCols(col):
    """Train and Predict for one event."""
    p = []
    for clf in clfs:
        # print 'trainPreprocessed:', trainPreprocessed, trainPreprocessed.shape
        # print 'labels_train[:, col]', labels_train[:, col], labels_train[:, col].shape
        clf.fit(trainPreprocessed, labels_train[:, col])
        p.append(clf.predict_proba(testPreprocessed)[:, 1])
    return p


yml = yaml.load(open(sys.argv[1]))

# Import package
for pkg, functions in yml['imports'].iteritems():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

# meta settings
fileName = yml['Meta']['file']
cores = yml['Meta']['cores']
subsample = yml['Meta']['subsample']
cache_preprocessed = yml['Meta']['cachePreprocessed']

if 'subsample_test' in yml['Meta'].keys():
    subsample_test = yml['Meta']['subsample_test']
else:
    subsample_test = 1

if 'addPreprocessed' in yml['Meta']:
    addPreprocessed = yml['Meta']['addPreprocessed']
else:
    addPreprocessed = []

# preprocessing pipeline
pipe = []
for item in yml['Preprocessing']:
    for method, params in item.iteritems():
        # pipe.append(_from_yaml_to_func(method, params))
# preprocess_base = make_pipeline(*pipe)

# post preprocessing
postpreprocess_base = None
if 'PostPreprocessing' in yml.keys():
    pipe = []
    for item in yml['PostPreprocessing']:
        for method, params in item.iteritems():
            # print 'method:', method
            # print 'params:', params
            # pipe.append(_from_yaml_to_func(method, params))

# models
clfs = []
for mdl in yml['Models']:
    clfs.append('Pipeline([ %s ])' % mdl)

# ## read arguments ###

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


print 'Running %s, to be saved in file %s' % (mode, fileName)

saveFolder = folder + fileName
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

# #### define lists #####
report = pd.DataFrame(index=[fileName])
# #### generate predictions #####

if not test:
    labels = np.load('../infos_val.npy')[:, :-1]

# ## AGGREGATE HERE
preds_tot = []

for i in range(len(clfs)):
    preds_tot.append([])
    for subject in subjects:
        preds_tot[i].append(np.load('%s/sub%d_clf%d.npy' % (saveFolder, subject, i)))
    preds_tot[i] = np.concatenate(preds_tot[i])
    # print 'preds_tot:', preds_tot, len(preds_tot), preds_tot[0].shape
    # print 'labels:', labels, len(labels), labels[0].shape
    if not test:
        auc = [roc_auc_score(trueVals, p) for trueVals, p in zip(labels[::subsample_test].T, preds_tot[i].T)]
        report['AUC'] = np.mean(auc)
        print np.mean(auc)

# ## save the model ###
np.save(folder + prefix + fileName + '.npy', preds_tot)

# ## save report
# report.to_csv("report/%s_%s.csv" % (prefix, fileName))
# print report
