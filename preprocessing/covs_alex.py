# -*- coding: utf-8 -*-
"""
Covariance model by alex.

@author: Alexandre Barachant
"""
import numpy as np
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_covariance
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import Pool
from functools import partial
from mne.event import _find_stim_steps

from eeg_config import N_EVENTS


def create_sequence(events):
    """create sequence from events.

    Create a sequence of non-overlapped States from labels.
    """
    # TODO: Make sure I have correctly understand what they did
    # init variable
    sequence = np.zeros((events.shape[1], 1))

    # get hand start
    no_vel = np.int64(_find_stim_steps(np.atleast_2d(events[0]), 0)[::2, 0])
    pos_vel = np.int64(_find_stim_steps(np.atleast_2d(events[1]), 0)[::2, 0])
    neg_vel = np.int64(_find_stim_steps(np.atleast_2d(events[2]), 0)[::2, 0])
    # print 'events', events, events.shape
    # print 'no_vel', no_vel, no_vel.shape
    # print 'pos_vel', pos_vel, pos_vel.shape
    # print 'neg_vel', neg_vel, neg_vel.shape

    sequence[events[0] == 1] = 0
    sequence[events[1] == 1] = 1
    sequence[events[2] == 1] = 2

    # for i in range(len(handStart)):
    #     j = 1
    #     sequence[(handStart[i] - 250):handStart[i]] = j
    #     j += 1
    #     sequence[handStart[i]:lift_on[i]] = j
    #     j += 1
    #     sequence[lift_on[i]:lift_off[i]] = j
    #     j += 1
    #     sequence[lift_off[i]:replace_on[i]] = j
    #     j += 1
    #     sequence[replace_on[i]:replace_off[i]] = j
    #     j += 1
    #     sequence[replace_off[i]:(replace_off[i] + 250)] = j
    #     j += 1

    return sequence


class DistanceCalculatorAlex(BaseEstimator, TransformerMixin):

    """Distance Calulator Based on MDM."""

    def __init__(self, metric_mean='logeuclid', metric_dist=['riemann'],
                 n_jobs=7, subsample=10):
        """Init."""
        self.metric_mean = metric_mean
        self.metric_dist = metric_dist
        self.n_jobs = n_jobs
        self.subsample = subsample

    def fit(self, X, y):
        """Fit."""
        self.mdm = MDM(metric=self.metric_mean, n_jobs=self.n_jobs)
        labels = np.squeeze(create_sequence(y.T)[::self.subsample])
        self.mdm.fit(X, labels)
        return self

    def transform(self, X, y=None):
        """Transform."""
        feattr = []
        for metric in self.metric_dist:
            self.mdm.metric_dist = metric
            feat = self.mdm.transform(X)
            # substract distance of the class 0
            feat = feat[:, 1:] - np.atleast_2d(feat[:, 0]).T
            feattr.append(feat)
        feattr = np.concatenate(feattr, axis=1)
        feattr[np.isnan(feattr)] = 0
        return feattr

    def fit_transform(self, X, y):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)


class DistanceCalculatorRafal(BaseEstimator, TransformerMixin):

    """Distance Calulator Based on MDM Rafal style."""

    def __init__(self, metric_mean='logeuclid', metric_dist=['riemann'],
                 n_jobs=12, subsample=10):
        """Init."""
        self.metric_mean = metric_mean
        self.metric_dist = metric_dist
        self.n_jobs = n_jobs
        self.subsample = subsample

    def fit(self, X, y):
        """Fit."""
        self.mdm = MDM(metric=self.metric_mean, n_jobs=self.n_jobs)
        labels = y[::self.subsample]
        pCalcMeans = partial(mean_covariance, metric=self.metric_mean)
        pool = Pool(processes=6)
        mc1 = pool.map(pCalcMeans, [X[labels[:, i] == 1] for i in range(N_EVENTS)])
        pool.close()
        pool = Pool(processes=6)
        mc0 = pool.map(pCalcMeans, [X[labels[:, i] == 0] for i in range(N_EVENTS)])
        pool.close()
        self.mdm.covmeans = mc1 + mc0
        return self

    def transform(self, X, y=None):
        """Transform."""
        feattr = []
        for metric in self.metric_dist:
            self.mdm.metric_dist = metric
            feat = self.mdm.transform(X)
            # print 'feat', feat, feat.shape
            # substract distance of the class 0
            feat = feat[:, 0:N_EVENTS] - feat[:, N_EVENTS:]
            feattr.append(feat)
        feattr = np.concatenate(feattr, axis=1)
        feattr[np.isnan(feattr)] = 0
        return feattr

    def fit_transform(self, X, y):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)
