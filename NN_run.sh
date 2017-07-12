#!/usr/bin/env sh
# Go throught all level by one script

module unload python
module load tensorflow/1.1.0
module load keras

n_subjects=1
# keras_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python
keras_python=python
workdir=./
# array=( val test )
array=( val )

cd $workdir
python genInfos.py --n_subjects=$n_subjects
rm -rf $workdir/lvl1/val/*
rm -rf $workdir/lvl1/report/*

for i in "${array[@]}"
do
    # Low pass EEG model x 2
    $keras_python lvl1/genPreds.py lvl1/models/FBL.yml $i --n_subjects=$n_subjects

    # NN models
    for filename in lvl1/models/NN*.yml; do
        $keras_python lvl1/genPreds_RNN.py $filename $i --n_subjects=$n_subjects
    done
    wait
done

# for i in "${array[@]}"
# do
#   for filename in lvl2/models/xgb_N*.yml; do
#     # filename='lvl2/models/xgb_NN_bags_model.yml'
#     # filename='lvl2/models/xgb_onlyNN.yml'
#     # filename='lvl2/models/xgb_NN_FBL.yml'
#     # filename='lvl2/models/xgb_NN_bags.yml'
#     echo "$filename"
# 
#     if [[ "$filename" == *"bags_model"* ]]
#     then
#       $keras_python lvl2/genEns_BagsModels.py $filename $i --n_subjects=$n_subjects -fast 1
#       # echo "pass genEns_BagsModels.py"
#     elif [[ "$filename" == *"bags"* ]]
#     then
#       $keras_python lvl2/genEns_BagsSubjects.py $filename $i --n_subjects=$n_subjects -fast 1
#       # echo "pass genEns_BagsSubjects.py"
#     else
#       $keras_python lvl2/genEns.py $filename $i --n_subjects=$n_subjects -fast 1
#       # echo "pass genEns.py"
#     fi
#   done
# done
# 
# $keras_python genFinal.py lvl3/models/Fast.yml
