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

import keras.backend as K

from preprocessing.aux import creat_mne_raw_object
from preprocessing.filterBank import FilterBank
from read_adapter import *
from eeg_config import CH_NAMES, N_EVENTS, subjects
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--n_subjects', type=int, action='store', default=2)
args, unknown = parser.parse_known_args()
subjects = range(1, args.n_subjects + 1)

####
yml = yaml.load(open(sys.argv[1]))
fileName = yml['Meta']['file']

filt2Dsize = yml['filt2Dsize'] if 'filt2Dsize' in yml.keys() else 0
filters = yml['filters']
delay = yml['delay']
skip = yml['skip']

if 'bags' in yml.keys():
    bags = yml['bags']
else:
    bags = 3

mode = sys.argv[2]
if mode == 'val':
    test = False
elif mode == 'test':
    test = True
else:
    raise('Invalid mode. Please specify either val or test')

###########
SUBJECTS = list(subjects)
TRAIN_SERIES = list(range(1, 5))
TEST_SERIES = [5]

N_ELECTRODES = 14

SAMPLE_SIZE = delay
DOWNSAMPLE = 1
TIME_POINTS = SAMPLE_SIZE // DOWNSAMPLE

START_TRAIN = delay

TRAIN_SIZE = 5120
# TRAIN_SIZE = 100

# We encapsulate the event / electrode data in a Source object.

def preprocessData(data):
    """Preprocess data with filterbank."""
    fb = FilterBank(filters)
    return fb.transform(data)

def one_hot_to_val(vec):
    n = vec.shape[0]
    return np.argmax(vec, axis=1)
    
def val_to_one_hot(vec, n_classes):
    n = vec.shape[0]
    res = np.zeros((n, n_classes))
    res[np.arange(n), vec] = 1
    return res

class Source:

    """Loads, preprocesses and holds data."""

    mean = None
    std = None

    def load_raw_data(self, subject, series):
        """
        Load data for a subject / series.
        n_points: int. The number of timepoints that can be predict/train. 
        Because the timepoints in the start are not valid for windows or there are no velocity.
        """
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
        self.n_points = self.data.shape[0] - START_TRAIN

        if not test:
            self.events = raw_train._data[14:].transpose()
        # print self.data.shape, self.events.shape
        # (num_time_points, num_ch), (num_time_point, num_labels)
            

    def normalize(self):
        """normalize data."""
        self.data -= self.mean
        self.data /= self.std

    def flow(self, batch_size=32, shuffle=True, seed=None):
        return NumpyArrayIterator(
            self.data, self.events, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)


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
            # The first few index shouldn't be counted because it doesn't have enough timepoints to construct a window
            if self.batch_index == 0:
                index_array = np.arange(n - START_TRAIN) + START_TRAIN
                if shuffle:
                    index_array = np.random.permutation(n-START_TRAIN) + START_TRAIN

            current_index = (self.batch_index * batch_size) % (n - START_TRAIN)
            if (n - START_TRAIN) > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = (n - START_TRAIN) - current_index
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
        source : nstance of `Source`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
    """

    def __init__(self, x, y, source,
                 batch_size=32, shuffle=False, seed=None):

        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x, dtype=K.floatx())

        '''
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        '''
        self.y = y
        self.source = source
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
        if filt2Dsize:
            input_shape = [current_batch_size, N_ELECTRODES, TIME_POINTS, 1]
        else:
            input_shape = [current_batch_size, N_ELECTRODES, TIME_POINTS]
        batch_x = np.zeros(input_shape, dtype=K.floatx())
        # Here should be a sliding window of size SAMPLE_SIZE
        # refer line 239 in genPreds_CNN_Tim.py
        for i, j in enumerate(index_array):
            sample = self.x[j-SAMPLE_SIZE:j]
            # Reverse so we get most recent point, otherwise downsampling drops
            # the last
            # DOWNSAMPLE-1 points.
            x = sample[::DOWNSAMPLE].T
            if filt2Dsize:
                batch_x[i, :, :, 0] = x
            else:
                batch_x[i, :, :] = x
        if self.y is None:
            return batch_x

        if self.y is None:
            return batch_x
        elif type(self.y) is list:
            batch_y = []
            for i in range(len(self.y)):
                batch_y.append(self.y[i][index_array])
        else:
            batch_y = self.y[index_array]

        # pdb.set_trace()
        return batch_x, batch_y

### End DataGenerator for Keras

# Lay out the Neural net.

import keras
from keras.layers import Input, merge, Dropout, Dense, Lambda, Flatten, Activation
from keras import layers
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

def create_net():

    dense = 512  # larger (1024 perhaps) would be better
    if filt2Dsize:
        input_shape=(N_ELECTRODES, SAMPLE_SIZE, 1)
        conv_layer = Conv2D(8, (N_ELECTRODES, filt2Dsize), name='conv2d_1')
    else:
        input_shape=(N_ELECTRODES, SAMPLE_SIZE)
        conv_layer = Conv1D(8, 1, name='conv1d_1')

    model = Sequential()
    model.add(Dropout(0.5, input_shape=input_shape))
    model.add(conv_layer)
    model.add(Flatten())
    model.add(Dense(dense, activation='relu', name='fc1'))
    model.add(Dropout(0.7))
    model.add(Dense(dense, activation='relu', name='fc2'))
    model.add(Dropout(0.7))
    model.add(Dense(N_EVENTS, activation='softmax', name='output'))
    
    optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    model.summary()
    return model

print 'Running in mode %s, saving to file %s' % (mode,fileName)
report = pd.DataFrame(index=[fileName])
start_time = time()

train_indices = np.zeros([TRAIN_SIZE], dtype=int) - 1

np.random.seed(67534)

BATCH_SIZE = 512
valid_series = [5]
max_epochs = 5

probs_bags = []
all_auc = []
# for bag in range(bags):
for bag in range(1):
    probs_tot = []
    lbls_tot = []
    for subject in subjects:
        # TODO: Also include the Nothing state to classification
        tseries = sorted(set(TRAIN_SERIES) - set(valid_series))
        train_source = TrainSource(subject, tseries)
        test_source = TestSource(subject, valid_series, train_source)

        # pdb.set_trace()
        model = create_net()
        model.fit_generator(train_source.flow(batch_size=BATCH_SIZE, shuffle=True),
                steps_per_epoch=train_source.n_points//BATCH_SIZE - 4,
                epochs=max_epochs,
                validation_data=test_source.flow(batch_size=BATCH_SIZE, shuffle=True),
                validation_steps=100,
            )
        probs = model.predict_generator(
                test_source.flow(batch_size=BATCH_SIZE, shuffle=False), 
                (test_source.n_points-1)//BATCH_SIZE + 1,
            )

        print probs.shape
        # Transform to one hot
        # probs = np.zeros((probs_val.shape[0], N_EVENTS))
        # probs = np.array[np.arange(probs.shape[0], probs_val)] = 1
        auc = np.mean([roc_auc_score(trueVals, p) for trueVals, p in 
                zip(test_source.events[START_TRAIN:, :].T, probs.T)])
        # pdb.set_trace()
        print 'Bag %d, subject %d, AUC: %.5f' % (bag, subject, auc)
        probs_tot.append(probs)
        lbls_tot.append(test_source.events[START_TRAIN:])

    probs_tot = np.concatenate(probs_tot)
    lbls_tot = np.concatenate(lbls_tot)
    auc = np.mean([roc_auc_score(trueVals, p) for trueVals, p in 
            zip(lbls_tot.transpose(), probs_tot.transpose())])
    all_auc.append(auc)
    probs_bags.append(probs_tot)

probs_bags = np.mean(probs_bags, axis=0)
np.save('val/val_%s.npy' % fileName, [probs_bags])

prefix = 'test_' if test else 'val_'
end_time = time()
report['Time'] = end_time - start_time
report['AUC'] = np.mean(all_auc)
report.to_csv("report/%s_%s.csv" % (prefix, fileName))
print report
