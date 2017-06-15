#!/usr/bin/env sh
# This script is written for test adn run the CNN modle in keras

module unload python
module load tensorflow/1.1.0
module load keras

n_subjects=10
gpu_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python

cd /home/lucien/eeg_mibk/Grasp-and-lift-EEG-challenge/
python genInfos.py --n_subjects $n_subjects
cd lvl1
cd val
find . -type f -not -name 'placeholder' -print0 | xargs -0 rm --
cd ..
cd report
find . -type f -not -name 'placeholder' -print0 | xargs -0 rm --
cd ..
$gpu_python genPreds_RNN.py models/RNN_FB_delay4000.yml val --n_subjects=$n_subjects
$gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz.yml val --n_subjects=$n_subjects
$gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz.yml val --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz_shorterDelay.yml val --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_30Hz.yml val --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_30Hz_shorterDelay.yml val --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_5Hz.yml val --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_7-30Hz.yml val --n_subjects=$n_subjects $i
