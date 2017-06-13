# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:28:32 2015.

Script written by Tim Hochberg with parameter tweaks by Bluefool.
https://www.kaggle.com/bitsofbits/grasp-and-lift-eeg-detection/naive-nnet 
Modifications: rc, alex
"""
import os
import sys
# if __name__ == '__main__' and __package__ is None:
if __package__ is None:
    filePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(filePath)

import yaml
import threading
from glob import glob
import numpy as np
import pandas as pd
from time import time

from sklearn.metrics import roc_auc_score

from mne import concatenate_raws, pick_types

from preprocessing.aux import creat_mne_raw_object
from preprocessing.filterBank import FilterBank
from read_adapter import *
from eeg_config import CH_NAMES, START_TRAIN, N_EVENTS, subjects
import argparse

import keras.backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--n_subjects', type=int, action='store', default=2)
args, unknown = parser.parse_known_args()
subjects = range(1, args.n_subjects + 1)

###########
SUBJECTS = list(subjects)
TRAIN_SERIES = list(range(1, 5))
TEST_SERIES = [5]

N_ELECTRODES = 14

SAMPLE_SIZE = delay
DOWNSAMPLE = 1
TIME_POINTS = SAMPLE_SIZE // DOWNSAMPLE

TRAIN_SIZE = 5120
# TRAIN_SIZE = 100

# We encapsulate the event / electrode data in a Source object.

def preprocessData(data):
    """Preprocess data with filterbank."""
    fb = FilterBank(filters)
    return fb.transform(data)

### From Keras ImageDataGenerator

class Iterator(object):
    """Abstract base class for image data iterators.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        '''
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        '''

        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'data format convention "' + data_format + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        '''
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        '''
        self.y = y
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x

        '''
        batch_y = self.y[index_array]
        '''
        batch_y = []
        for i in range(len(self.y)):
            batch_y.append(self.y[i][index_array])

        return batch_x, batch_y


### End DataGenerator for Keras

class Source:

    """Loads, preprocesses and holds data."""

    mean = None
    std = None

    def load_raw_data(self, subject, series):
        """Load data for a subject / series."""
        # test = series == TEST_SERIES
        test = False
        if not test:
            fnames = [glob(get_horizo_path(subject, i)) for i in series]
        else:
            fnames = [glob('../data/test/subj%d_series%d_data.csv' %
                      (subject, i)) for i in series]
        fnames = list(np.concatenate(fnames))
        fnames.sort()
        self.fnames = fnames
        action_1D_type = 'HO'
        raw_train = [creat_mne_raw_object(fnames[i], i, read_events=action_1D_type) 
                for i in range(len(fnames))]
        raw_train = concatenate_raws(raw_train)
        # pick eeg signal
        picks = pick_types(raw_train.info, eeg=True)

        self.data = raw_train._data[picks].transpose()

        self.data = preprocessData(self.data)

        if not test:
            self.events = raw_train._data[14:].transpose()
        # print self.data.shape, self.events.shape
        # (num_time_points, num_ch), (num_time_point, num_labels)
            

    def normalize(self):
        """normalize data."""
        self.data -= self.mean
        self.data /= self.std


class TrainSource(Source):

    """Source for training data."""

    def __init__(self, subject, series_list):
        """Init."""
        self.load_raw_data(subject, series_list)
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.normalize()


# Note that Test/Submit sources use the mean/std from the training data.
# This is both standard practice and avoids using future data in theano
# test set.

class TestSource(Source):

    """Source for test data."""

    def __init__(self, subject, series, train_source):
        """Init."""
        self.load_raw_data(subject, series)
        self.mean = train_source.mean
        self.std = train_source.std
        self.normalize()


# Lay out the Neural net.


# Do the training.
print 'Running in mode %s, saving to file %s' % (mode,fileName)
report = pd.DataFrame(index=[fileName])
start_time = time()

train_indices = np.zeros([TRAIN_SIZE], dtype=int) - 1

np.random.seed(67534)

valid_series = [5]
max_epochs = 100

probs_bags = []
for bag in range(bags):
probs_tot = []
lbls_tot = []
for subject in range(1, 13):
    # TODO: Also include the Nothing state to classification
    tseries = sorted(set(TRAIN_SERIES) - set(valid_series))
    train_source = TrainSource(subject, tseries)
    test_source = TestSource(subject, valid_series, train_source)
    net = create_net(train_source, test_source, max_epochs=max_epochs,
                        train_val_split=False)
    dummy = net.fit(train_indices, train_indices)
    indices = np.arange(START_TRAIN, len(test_source.data))
    probs = net.predict_proba(indices)
    auc = np.mean([roc_auc_score(trueVals, p) for trueVals, p in 
            zip(test_source.events[START_TRAIN:].T[1:, :], probs.T[1:, :])])
    print 'Bag %d, subject %d, AUC: %.5f' % (bag, subject, auc)
    probs_tot.append(probs)
    lbls_tot.append(test_source.events[START_TRAIN:])

probs_tot = np.concatenate(probs_tot)
lbls_tot = np.concatenate(lbls_tot)
auc = np.mean([roc_auc_score(trueVals, p) for trueVals, p in 
zip(lbls_tot.transpose(), probs_tot.transpose())])
probs_bags.append(probs_tot)

probs_bags = np.mean(probs_bags, axis=0)
np.save('val/val_%s.npy' % fileName, [probs_bags])


prefix = 'test_' if test else 'val_'
end_time = time()
report['Time'] = end_time - start_time
report.to_csv("report/%s_%s.csv" % (prefix, fileName))
print report
