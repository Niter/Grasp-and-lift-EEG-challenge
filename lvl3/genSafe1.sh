#!/usr/bin/env sh

module unload python
module load tensorflow
module load keras

keras_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python
workdir=/home/lucien/eeg_mibk/Grasp-and-lift-EEG-challenge
n_subjects=4

cd $workdir/lvl3
$keras_python genFinal.py models/Safe1.yml
