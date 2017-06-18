# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:35:28 2015.

@author: fornax
"""
import numpy as np
import pandas as pd
from glob import glob
from mne import concatenate_raws

import pdb
from preprocessing.aux import creat_mne_raw_object
from eeg_config import CH_NAMES
from read_adapter import get_all_horizon_path_from_the_subject
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_subjects', type=int, action='store', default=2)
args, unknown = parser.parse_known_args()
subjects = range(1, args.n_subjects + 1)

# #### define lists #####

lbls_tot = []
subjects_val_tot = []
series_val_tot = []

ids_tot = []
subjects_test_tot = []
series_test_tot = []

# #### generate predictions #####
for subject in subjects:
    print 'Loading data for subject %d...' % subject
    # ############### READ DATA ###############################################
    # fnames = glob('data/train/subj%d_series*_data.csv' % (subject))
    fnames = glob(get_all_horizon_path_from_the_subject(subject))
    fnames.sort()
    fnames_val = fnames[3:5]

    fnames_test = fnames[-1:]
    # fnames_test = glob('data/test/subj%d_series*_data.csv' % (subject))
    # fnames_test.sort()
    val_index_offset = 3
    test_index_offset = 4

    # Note that the 2nd args of creat_mne_raw_object is zero-based
    action_1D_type = 'HO'
    raw_val = concatenate_raws([creat_mne_raw_object(fname, val_index_offset + i, read_events=action_1D_type) 
            for i, fname in enumerate(fnames_val)])
    raw_test = concatenate_raws([creat_mne_raw_object(fname, test_index_offset + i, read_events=action_1D_type) 
            for i, fname in enumerate(fnames_test)])

    # extract labels for series 7&8
    labels = raw_val._data[len(CH_NAMES):]
    lbls_tot.append(labels.transpose())

    # aggregate infos for validation (series 7&8)
    raw_series3 = creat_mne_raw_object(fnames_val[0], 3, action_1D_type)
    raw_series4 = creat_mne_raw_object(fnames_val[1], 4, action_1D_type)
    series = np.array([4] * raw_series3.n_times + [5] * raw_series4.n_times)
    series_val_tot.append(series)

    subjs = np.array([subject]*labels.shape[1])
    subjects_val_tot.append(subjs)

    # aggregate infos for test (series 9&10)
    # ids are the name/idx of the timepoints
    raw_series5 = creat_mne_raw_object(fnames_test[0], 4, read_events=action_1D_type)
    series = np.array([5] * raw_series5.n_times)
    ids = np.concatenate([np.arange(raw_series5.n_times) for fname in fnames_test])
    ids_tot.append(ids)
    series_test_tot.append(series)

    subjs = np.array([subject]*raw_test.n_times)
    subjects_test_tot.append(subjs)


# save validation infos
subjects_val_tot = np.concatenate(subjects_val_tot)
series_val_tot = np.concatenate(series_val_tot)
lbls_tot = np.concatenate(lbls_tot)
toSave = np.c_[lbls_tot, subjects_val_tot, series_val_tot]
np.save('infos_val.npy', toSave)

# save test infos
subjects_test_tot = np.concatenate(subjects_test_tot)
series_test_tot = np.concatenate(series_test_tot)
ids_tot = np.concatenate(ids_tot)
toSave = np.c_[ids_tot, subjects_test_tot, series_test_tot]
np.save('infos_test.npy', toSave)
