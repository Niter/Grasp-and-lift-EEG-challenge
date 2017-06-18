#!/usr/bin/env sh
# This scrip is to go through all models that are configured in lvl1/models
# And store the predictions in lvl1/val and the report in lvl1/reprot

module unload python
module load tensorflow/1.1.0
module load keras

n_subjects=12
gpu_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python
workdir=/home/lucien/eeg_mibk/Grasp-and-lift-EEG-challenge
# array=( val test )
array=( val )

cd $workdir
python genInfos.py --n_subjects=$n_subjects
# find $workdir/lvl1/val/ -type f -name "*" -print0 | xargs -0 rm --
# rm -rf $workdir/lvl1/val/*
# find $workdir/lvl1/report/ -type f -name '*.csv' -print0 | xargs -0 rm --
# rm -rf $workdir/lvl1/report/*

cd $workdir/lvl1
for i in "${array[@]}"
do
    # generate validation preds
    # cov models x 7
    $gpu_python genPreds.py models/CovAlex_500.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovAlex_500_1-15.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovAlex_500_20-35.yml $i --n_subjects=$n_subjects &
    # $gpu_python genPreds.py models/CovAlex_500_70-150.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovAlex_250_35Hz.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovAlex_500_35Hz.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovERP_dist.yml $i --n_subjects=$n_subjects &
    wait

    # PolynomialFeatures cov model x 7
    $gpu_python genPreds.py models/CovAlex_500_poly.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovAlex_500_1-15_poly.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovAlex_500_20-35_poly.yml $i --n_subjects=$n_subjects &
    # $gpu_python genPreds.py models/CovAlex_500_70-150_poly.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovAlex_250_35Hz_poly.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovAlex_500_35Hz_poly.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovERP_dist_poly.yml $i --n_subjects=$n_subjects &
    wait

    # rafal cov model x 2
    $gpu_python genPreds.py models/CovRafal_256_35Hz.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovRafal_500_35Hz.yml $i --n_subjects=$n_subjects &
    wait

    # aggregated cov model x 2
    $gpu_python genPreds.py models/CovAlex_All.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/CovAlex_old_All.yml $i --n_subjects=$n_subjects &
    wait

    # Low pass EEG model x 2
    $gpu_python genPreds.py models/FBL.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/FBL_delay.yml $i --n_subjects=$n_subjects &

    # Hybrid model (cov + FBL) x 3
    $gpu_python genPreds.py models/FBLC_256pts.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/FBLCR_256.yml $i --n_subjects=$n_subjects &
    $gpu_python genPreds.py models/FBLCR_All.yml $i --n_subjects=$n_subjects &

    # NN models
    $gpu_python genPreds_RNN.py models/NN_16.yml val --n_subjects=$n_subjects
    $gpu_python genPreds_RNN.py models/NN_32.yml val --n_subjects=$n_subjects
    $gpu_python genPreds_RNN.py models/NN_64.yml val --n_subjects=$n_subjects
    $gpu_python genPreds_RNN.py models/NN_128.yml $i --n_subjects=$n_subjects
    $gpu_python genPreds_RNN.py models/NN_256.yml $i --n_subjects=$n_subjects
    $gpu_python genPreds_RNN.py models/NN_512.yml $i --n_subjects=$n_subjects
    $gpu_python genPreds_RNN.py models/RNN_FB_delay4000.yml $i --n_subjects=$n_subjects
    # $gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz.yml $i --n_subjects=$n_subjects
    # $gpu_python genPreds_KerasCNN.py models/cnn_script_2D_30Hz_shorterDelay.yml $i --n_subjects=$n_subjects
    # $gpu_python genPreds_KerasCNN.py models/cnn_script_1D_30Hz.yml $i --n_subjects=$n_subjects
    # $gpu_python genPreds_KerasCNN.py models/cnn_script_1D_30Hz_shorterDelay.yml $i --n_subjects=$n_subjects
    # $gpu_python genPreds_KerasCNN.py models/cnn_script_1D_5Hz.yml $i --n_subjects=$n_subjects
    # $gpu_python genPreds_KerasCNN.py models/cnn_script_1D_7-30Hz.yml $i --n_subjects=$n_subjects
    wait
done
