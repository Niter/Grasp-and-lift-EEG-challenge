#!/usr/bin/env sh
# Go throught all level by one script

module unload python
module load tensorflow/1.1.0
module load keras

n_subjects=1
keras_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python
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
    # $keras_python genPreds.py models/CovAlex_250_35Hz.yml $i --n_subjects=$n_subjects

    # Low pass EEG model x 2
    # $keras_python genPreds.py models/FBL.yml $i --n_subjects=$n_subjects &

    # NN models
    for filename in models/RNN*.yml; do
        $keras_python genPreds_RNN.py $filename $i --n_subjects=$n_subjects
    done
    # $keras_python genPreds_RNN.py models/RNN_FBL_PCA_128.yml $i --n_subjects=$n_subjects
    # $keras_python genPreds_RNN.py models/RNN_FB_delay4000.yml $i --n_subjects=$n_subjects
    # $keras_python genPreds_RNN.py models/NN_16.yml val --n_subjects=$n_subjects
    # $keras_python genPreds_RNN.py models/NN_32.yml val --n_subjects=$n_subjects
    # $keras_python genPreds_RNN.py models/NN_64.yml val --n_subjects=$n_subjects
    # $keras_python genPreds_RNN.py models/NN_128.yml $i --n_subjects=$n_subjects
    # $keras_python genPreds_RNN.py models/NN_256.yml $i --n_subjects=$n_subjects
    # $keras_python genPreds_RNN.py models/NN_512.yml $i --n_subjects=$n_subjects
    wait
done

# cd $workdir/lvl2
# for i in "${array[@]}"
# do
#   for filename in models/xgb_NN_FBL*.yml; do
#     echo "$filename"
# 
#     if [[ "$filename" == *"bags_model"* ]]
#     then
#       $keras_python genEns_BagsModels.py $filename $i --n_subjects=$n_subjects -fast 1
#       # echo "pass genEns_BagsModels.py"
#     elif [[ "$filename" == *"bags"* ]]
#     then
#       $keras_python genEns_BagsSubjects.py $filename $i --n_subjects=$n_subjects -fast 1
#       # echo "pass genEns_BagsSubjects.py"
#     else
#       $keras_python genEns.py $filename $i --n_subjects=$n_subjects -fast 1
#       # echo "pass genEns.py"
#     fi
#   done
# done
# 
# cd $workdir/lvl3
# $keras_python genFinal.py models/Fast.yml
