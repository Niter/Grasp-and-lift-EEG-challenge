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
# rm -rf $workdir/lvl1/val/*
# rm -rf $workdir/lvl1/report/*

# cd $workdir/lvl1
# for i in "${array[@]}"
# do
#     # Low pass EEG model x 2
#     $keras_python genPreds.py models/FBL.yml $i --n_subjects=$n_subjects &
# 
#     # NN models
#     for filename in models/NN*.yml; do
#         $keras_python genPreds_RNN.py $filename $i --n_subjects=$n_subjects
#     done
#     wait
# done

cd $workdir/lvl2
for i in "${array[@]}"
do
  for filename in models/xgb_NN*.yml; do
    # filename='models/xgb_NN_bags_model.yml'
    # filename='models/xgb_onlyNN.yml'
    # filename='models/xgb_NN_FBL.yml'
    # filename='models/xgb_NN_bags.yml'
    echo "$filename"

    if [[ "$filename" == *"bags_model"* ]]
    then
      $keras_python genEns_BagsModels.py $filename $i --n_subjects=$n_subjects -fast 1
      # echo "pass genEns_BagsModels.py"
    elif [[ "$filename" == *"bags"* ]]
    then
      $keras_python genEns_BagsSubjects.py $filename $i --n_subjects=$n_subjects -fast 1
      # echo "pass genEns_BagsSubjects.py"
    else
      $keras_python genEns.py $filename $i --n_subjects=$n_subjects -fast 1
      # echo "pass genEns.py"
    fi
  done
done

cd $workdir/lvl3
$keras_python genFinal.py models/Fast.yml
