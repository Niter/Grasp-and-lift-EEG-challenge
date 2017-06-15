#!/usr/bin/env sh
# This script is written for test adn run the CNN modle in keras

module unload python
module load tensorflow/1.1.0
module load keras

n_subjects=2
gpu_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python

cd ../
python genInfos.py --n_subjects $n_subjects
cd lvl1
rm -rf val/*
# $gpu_python genPreds_RNN.py models/RNN_FB_delay4000.yml val --n_subjects=$n_subjects
$gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz.yml val --n_subjects=$n_subjects
$gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz.yml --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz_shorterDelay.yml --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_30Hz.yml --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_30Hz_shorterDelay.yml --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_5Hz.yml --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_7-30Hz.yml --n_subjects=$n_subjects $i
