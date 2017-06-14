#!/usr/bin/env sh
# This script is written for test adn run the CNN modle in keras

n_subjects=2
cd ../
python genInfos.py --n_subjects $n_subjects
cd lvl1
rm -rf val/*
# /opt/packages/keras/keras_2.0.4/kerasEnv/bin/python genPreds_RNN.py models/RNN_FB_delay4000.yml val --n_subjects=$n_subjects
/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python genPreds_KerasCNN.py models/cnn_script_2D_30Hz.yml val --n_subjects=$n_subjects
