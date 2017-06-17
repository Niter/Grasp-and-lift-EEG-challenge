#!/usr/bin/env sh
# This scrip is to go through all models that are configured in lvl1/models
# And store the predictions in lvl1/val and the report in lvl1/reprot

module unload python
module load tensorflow
module load keras

keras_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python
workdir=/home/lucien/eeg_mibk/Grasp-and-lift-EEG-challenge
n_subjects=4

cd $workdir
# python genInfos.py --n_subjects=$n_subjects
cd $workdir/lvl2

# for filename in models/*.yml; do
# filename="models/CNN_196.yml"
# filename="models/RNN_256_customDelay_allModels_ADAM_bags_model.yml"
# filename="models/RNN_256_customDelay_allModels_ADAM.yml"
# filename="models/RNN_256_delay4000_allModels_ADAM_bags.yml"
# filename='models/xgb_bags.yml'
# filename="models/xgb_longshort_bags_model.yml"
filename="models/NN_256_allModels_ADAM_bags.yml"
echo "$filename"
if [[ "$filename" == *"bags_model"* ]]
then
    $keras_python genEns_BagsModels.py $filename val --n_subjects=$n_subjects
    # echo "pass genEns_BagsModels.py"
elif [[ "$filename" == *"bags"* ]]
then
    $keras_python genEns_BagsSubjects.py $filename val --n_subjects=$n_subjects
    # echo "pass genEns_BagsSubjects.py"
else
    $keras_python genEns.py $filename val --n_subjects=$n_subjects
    # echo "pass genEns.py"
fi
# done
