# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 22:00:08 2015.

@author: rc, alexandre
"""
import os
import sys
import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.channels import read_montage
from mne import create_info, concatenate_raws, pick_types
from sklearn.base import BaseEstimator, TransformerMixin
from glob import glob

import pdb
from eeg_config import CH_NAMES
from read_adapter import get_all_horizon_path_from_the_subject, get_horizo_velocity, get_vertic_velocity


def getChannelNames():
    """Return Channels names."""
    # return ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
    #         'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2',
    #         'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
    #         'O2', 'PO10']
    return CH_NAMES


def getEventNames():
    """Return Event name."""
    # return ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff',
    #         'Replace', 'BothReleased']
    return ['no_vel', 'pos_vel', 'neg_vel']


def load_raw_data(subject, test=False):
    """Load Raw data from files.

    For a given subject, csv files are loaded, converted to MNE raw instance
    and concatenated.
    If test is True, training data are composed of series 1 to 8 and test data
    of series 9 and test. Otherwise, training data are series 1 to 6 and test
    data series 7 and 8.
    """
    # fnames_train = glob('../data/train/subj%d_series*_data.csv' % (subject))
    fnames_train = glob(get_all_horizon_path_from_the_subject(subject))
    fnames_train.sort()
    if test:
        fnames_test = fnames_train[-1:]
        fnames_train = fnames_train[:-1]
        test_idx_offset = 4
    else:
        fnames_test = fnames_train[2:4]
        fnames_train = fnames_train[:2]
        test_idx_offset = 2

    # read and concatenate all the files
    action_1D_type = 'HO'
    raw_train = [creat_mne_raw_object(fname, i, read_events=action_1D_type) for i, fname in enumerate(fnames_train)]
    raw_train = concatenate_raws(raw_train)
    # pick eeg signal
    picks = pick_types(raw_train.info, eeg=True)

    # get training data
    data_train = raw_train._data[picks].T
    labels_train = raw_train._data[len(CH_NAMES):].T

    print 'fnames_train:', fnames_train
    print 'fnames_test:', fnames_test
    raw_test = [creat_mne_raw_object(fname, test_idx_offset + i, read_events=action_1D_type) 
            for i, fname in enumerate(fnames_test)]
    raw_test = concatenate_raws(raw_test)
    data_test = raw_test._data[picks].T

    # extract labels if validating on series 7&8
    labels_test = None
    if not test:
        labels_test = raw_test._data[len(CH_NAMES):].T

    return data_train, labels_train, data_test, labels_test


def creat_mne_raw_object(fname, idx_subject, read_events='HO'):
    """
    Create a mne raw instance from csv file.

    idx_subject is zero-based index
    """
    # Read EEG file
    # data = pd.read_csv(fname)
    data = np.loadtxt(fname, delimiter=',')
    data = data.T[2:, :]

    # get chanel names
    # ch_names = list(data.columns[1:])
    ch_names = CH_NAMES

    # read EEG standard montage from mne
    montage = read_montage('standard_1005', ch_names)

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[:])

    if read_events:
        # events file
        ev_fname = fname.replace('_data', '_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = ['Nothing', 'Positive', 'Negative']
        # events_data = np.array(events[events_names]).T
        events_data = get_horizo_velocity() if read_events == 'HO' else \
                      get_vertic_velocity() if read_events == 'VE' else None
        # pdb.set_trace()

        events_data = events_data.T[idx_subject, :]
        events = np.zeros([3, events_data.shape[0]])
        events[0, events_data == 0] = 1
        events[1, events_data > 0] = 1
        events[2, events_data < 0] = 1

        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*3)
        # ch_names.extend(events_names)
        ch_names = ch_names + events_names
        # concatenate event file and data
        data = np.concatenate((data, events))

    # create and populate MNE info structure

    info = create_info(ch_names, sfreq=128.0, ch_types=ch_type,
                       montage=montage)
    info['experimento'] = fname

    # create raw object
    raw = RawArray(data, info, verbose=False)

    return raw


def sliding_window(sig, window=512, subsample=10, estimator=None):
    """Extract a slinding window from signal.

    Raw signal is padded with zeros on the left to avoid use of future data.
    """
    Ne, Ns = sig.shape
    # get the index before padding
    ix = range(0, Ns, subsample)

    # padd data
    padd = np.zeros((Ne, int(window) - 1))
    sig = np.concatenate((padd, sig), axis=1)
    Ne, Ns = sig.shape

    if estimator is None:
        estimator = np.array
    # call this to get the shape
    X = estimator(sig[:, 0:window])
    dims = list(X.shape)
    dims.insert(0, len(ix))
    dims = tuple(dims)

    # allocate array
    X = np.empty(dims, dtype=X.dtype)
    for i in range(len(ix)):
        X[i] = estimator(sig[:, ix[i]:(ix[i] + window)])

    return X


def delay_preds(X, delay=100, skip=2, subsample=1, start=0, jump=None):
    """Delay predictions.

    Create a feature vector by concatenation of present and past sample.
    The concatenation is done by shifting data to the right :

    out = | x1 x2 x3 ...  xn   |
          | 0  x1 x2 ...  xn-1 |
          | 0  0  x1 ...  xn-2 |

    No use of future data.
    """
    if jump is None:
        jump = range(0, delay, skip)
    Ns, Ne = X.shape
    # print 'Ns:', Ns
    # print 'Ne:', Ne
    Ns_subsampled = len(range(start, Ns, subsample))
    out = np.zeros((Ns_subsampled, Ne * len(jump)))
    for i, sk in enumerate(jump):
        chunk = X[0:(Ns - sk)][start::subsample]
        out[(Ns_subsampled-chunk.shape[0]):, (i * Ne):((i + 1) * Ne)] = chunk
    # print 'out:', out, out.shape
    # print 'delay:', delay
    # print 'skip:', skip
    # print 'subsample:', subsample
    return out


def delay_preds_2d(X, delay=100, skip=2, subsample=1, start=0, jump=None):
    """Delay predictions with 2d shape.

    Same thing as delay_pred, but return delayed prediction with a 2d shape.
    """
    if jump is None:
        jump = range(0, delay, skip)
    Ns, Ne = X.shape
    Ns_subsampled = len(range(start, Ns, subsample))
    out = np.zeros((Ns_subsampled, len(jump), Ne))
    for i, sk in enumerate(jump):
        chunk = X[0:(Ns - sk)][start::subsample]
        out[(Ns_subsampled-chunk.shape[0]):, i, :] = chunk
    return out[:, ::-1, :]


class SlidingWindow(BaseEstimator, TransformerMixin):

    """Sliding Window tranformer Mixin."""

    def __init__(self, window=100, subsample=1, estimator=np.array):
        """Init."""
        self.window = window
        self.subsample = subsample
        self.estimator = estimator

    def fit(self, X, y=None):
        """Fit, not used."""
        return self

    def transform(self, X, y=None):
        """Transform."""
        return sliding_window(X.T, window=self.window,
                              subsample=self.subsample,
                              estimator=self.estimator)

    def update_subsample(self, old_sub, new_sub):
        """update subsampling."""
        self.subsample = new_sub


class SubSample(BaseEstimator, TransformerMixin):

    """Subsample tranformer Mixin."""

    def __init__(self, subsample=10):
        """Init."""
        self.subsample = subsample

    def fit(self, X, y=None):
        """Fit, not used."""
        return self

    def transform(self, X, y=None):
        """Transform."""
        return X[::self.subsample]

    def update_subsample(self, old_sub, new_sub):
        """update subsampling."""
        self.subsample = new_sub


class DelayPreds(BaseEstimator, TransformerMixin):

    """Delayed prediction tranformer Mixin."""

    def __init__(self, delay=1000, skip=100, two_dim=False):
        """Init."""
        self.delay = delay
        self.skip = skip
        self.two_dim = two_dim

    def fit(self, X, y=None):
        """Fit, not used."""
        return self

    def transform(self, X, y=None):
        """Transform."""
        if self.two_dim:
            return delay_preds_2d(X, delay=self.delay, skip=self.skip)
        else:
            return delay_preds(X, delay=self.delay, skip=self.skip)

    def update_subsample(self, old_sub, new_sub):
        """update subsampling."""
        ratio = old_sub / new_sub
        self.delay = int(self.delay * ratio)
        self.skip = int(self.skip * ratio)


class NoneTransformer(BaseEstimator, TransformerMixin):

    """Return None Transformer."""

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None):
        """Fit, not used."""
        return self

    def transform(self, X, y=None):
        """Transform."""
        return None
