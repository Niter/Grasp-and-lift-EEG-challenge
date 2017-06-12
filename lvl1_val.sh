#!/usr/bin/env sh

python genInfos.py
cd lvl1
rm -rf val/*
python genPreds.py models/CovERP_dist.yml val
