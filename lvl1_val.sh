#!/usr/bin/env sh

python genInfos.py
cd lvl1
rm -rf val/*
python genPreds.py models/CovAlex_500_1-15.yml val
