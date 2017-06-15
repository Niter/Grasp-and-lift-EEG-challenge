#!/usr/bin/env sh
# This scrip is to go through all models that are configured in lvl1/models
# And store the predictions in lvl1/val and the report in lvl1/reprot

module unload python
module load tensorflow/1.1.0
module load keras

n_subjects=10
gpu_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python

cd ..
python genInfos.py --n_subjects=$n_subjects
cd lvl1
cd val
find . -type f -not -name 'placeholder' -print0 | xargs -0 rm --
cd ..
cd report
find . -type f -not -name 'placeholder' -print0 | xargs -0 rm --
cd ..

# generate validation preds
# cov models
$gpu_python genPreds.py models/CovAlex_500.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_500_1-15.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_500_20-35.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_500_70-150.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_250_35Hz.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_500_35Hz.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovERP_dist.yml val --n_subjects=$n_subjects &
wait
# PolynomialFeatures cov model
$gpu_python genPreds.py models/CovAlex_500_poly.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_500_1-15_poly.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_500_20-35_poly.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_500_70-150_poly.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_250_35Hz_poly.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_500_35Hz_poly.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovERP_dist_poly.yml val --n_subjects=$n_subjects &
wait
# rafal cov model 
$gpu_python genPreds.py models/CovRafal_256_35Hz.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovRafal_500_35Hz.yml val --n_subjects=$n_subjects &
wait
# aggregated cov model
$gpu_python genPreds.py models/CovAlex_All.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/CovAlex_old_All.yml val --n_subjects=$n_subjects &
wait

# Low pass EEG model
$gpu_python genPreds.py models/FBL.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/FBL_delay.yml val --n_subjects=$n_subjects &
wait

# Hybrid model (cov + FBL)
$gpu_python genPreds.py models/FBLC_256pts.yml val --n_subjects=$n_subjects &
$gpu_python genPreds.py models/FBLCR_256.yml val --n_subjects=$n_subjects &
wait

# NN models
$gpu_python genPreds_RNN.py models/RNN_FB_delay4000.yml val --n_subjects=$n_subjects
$gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz.yml val --n_subjects=$n_subjects
$gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz.yml val --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz_shorterDelay.yml val --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_30Hz.yml val --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_30Hz_shorterDelay.yml val --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_5Hz.yml val --n_subjects=$n_subjects $i
$gpu_python genPreds_KerasCNN.py models/cnn_script_1D_7-30Hz.yml val --n_subjects=$n_subjects $i
