#!/usr/bin/env sh

python genInfos.py
cd lvl1
rm -rf val/*
python genPreds_RNN.py models/RNN_FB_delay4000.yml val
