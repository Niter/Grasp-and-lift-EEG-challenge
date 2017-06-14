#!/usr/bin/env sh

n_subjects=10
cd ../
python genInfos.py --n_subjects $n_subjects
cd lvl1
rm -rf val/*
# /opt/packages/keras/keras_2.0.4/kerasEnv/bin/python genPreds_RNN.py models/RNN_FB_delay4000.yml val --n_subjects=$n_subjects
python genPreds_CNN_Tim.py models/cnn_script_2D_30Hz.yml val --n_subjects=$n_subjects
